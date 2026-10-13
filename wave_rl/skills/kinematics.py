import numpy as np
import torch


def quat_to_rotation(q) -> np.ndarray:
    """(4,) wxyz -> (3, 3). Construction-time only, so numpy and no batching."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def homogeneous(rotation, translation) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def skew(v) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class JointChain:
    """The serial kinematic chain from the world to a robot's end-effector frame, as batched
    torch forward kinematics and the geometric twist Jacobian that goes with it.

    WHY THIS EXISTS. An arm's action channels are JOINT targets, so which joint motion leaves
    the tool where it is changes with the configuration. `J(q)` answers it: its null space is
    the self-motion of a redundant arm, the one direction the warp can move the joints along
    without moving the tool. It is a PLANT property and carries no task knowledge.

    THE FORM IS CHOSEN FOR KERNEL COUNT, NOT ELEGANCE, AND THAT IS LOAD-BEARING. The
    obvious transcription — accumulate a quaternion and a position, one body at a time, with
    `quat_mul` / `quat_rotate` — is correct but costs about a whole control tick per call at
    2048 scenes, all of it launch overhead: roughly 250 tiny kernels from a Python loop, which
    does not improve on a faster card. Two things avoid it and neither costs accuracy:

    1. **Fixed transforms are COLLAPSED at construction.** Every jointless carrier frame on
       the path (`panda_hand_tip` is one, and it holds the offset that puts the frame between
       the pads) is folded into its neighbours, so the loop runs once per HINGE — seven
       iterations, not twelve.
    2. **A hinge is AFFINE in (sin, 1 - cos).** Rodrigues gives
       ``R(t) = I + sin(t) K + (1 - cos(t)) K^2`` about a constant local axis, and the
       surrounding fixed transforms are constants, so the whole 4x4 joint transform is
       ``A + sin(t) B + (1 - cos(t)) C`` with A, B, C precomputed. All seven are therefore
       built for the whole batch in FOUR kernels, and the only sequential work left is seven
       4x4 batched matmuls.

    The transcription is MuJoCo's own composition (`mj_kinematics` / `mj_jacBody`).
    """

    def __init__(self, steps: list[dict], tip_transform, joint_columns: list[int],
                 action_scale: float, device) -> None:
        self.n_joints = len(steps)
        self.joint_columns = torch.as_tensor(joint_columns, dtype=torch.long, device=device)
        self.action_scale = action_scale
        self.device = device
        self.names = [s["name"] for s in steps]

        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
        #: (n, 4, 4) the fixed transform from the previous hinge's frame to this hinge's body
        self.fixed = t([s["fixed"] for s in steps])
        #: (n, 4, 4) each the constant term of `A + sin B + (1-cos) C` for this hinge
        self.rot_a = t([s["A"] for s in steps])
        self.rot_b = t([s["B"] for s in steps])
        self.rot_c = t([s["C"] for s in steps])
        #: (n, 3) the hinge axis and anchor in the frame `fixed` lands in
        self.axis_local = t([s["axis"] for s in steps])
        self.anchor_local = t([s["anchor"] for s in steps])
        self.qpos0 = t([s["qpos0"] for s in steps])
        #: (4, 4) the last hinge's frame to the end-effector frame
        self.tip = t(tip_transform)

    def banner(self) -> str:
        return (f"== KINEMATIC CHAIN  {self.n_joints} hinges -> {self.names[-1]!r} | "
                f"joints {self.names} | robot_qpos columns "
                f"{self.joint_columns.tolist()} | action scale {self.action_scale:.4g} "
                f"rad per unit command")

    def forward(self, q: torch.Tensor):
        """``q`` (B, n_joints) in radians -> (tip (B, 3), axes (B, n, 3), anchors (B, n, 3)) in
        the world frame the simulator uses."""
        B = q.shape[0]
        angle = q - self.qpos0
        # every hinge's 4x4, for the whole batch, in one shot
        sin = torch.sin(angle)[..., None, None]
        cos1 = (1.0 - torch.cos(angle))[..., None, None]
        joint = self.rot_a + sin * self.rot_b + cos1 * self.rot_c        # (B, n, 4, 4)

        transform = torch.eye(4, device=q.device, dtype=q.dtype).expand(B, 4, 4)
        axes, anchors = [], []
        for k in range(self.n_joints):
            pre = transform @ self.fixed[k]                              # (B, 4, 4)
            rotation = pre[:, :3, :3]
            axes.append(rotation @ self.axis_local[k])
            anchors.append(rotation @ self.anchor_local[k] + pre[:, :3, 3])
            transform = pre @ joint[:, k]
        tip = (transform @ self.tip)[:, :3, 3]
        return tip, torch.stack(axes, dim=1), torch.stack(anchors, dim=1)

    def reach(self) -> float:
        """An upper bound on how far the end-effector frame can get from the chain's root,
        in metres: every fixed link offset laid end to end plus the tool offset. Derived from
        the compiled model, so it is the arm's own number, never typed."""
        links = self.fixed[:, :3, 3].norm(dim=-1).sum() + self.tip[:3, 3].norm()
        return float(links)

    def twist_jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """(B, 6, n_joints) — rows 0:3 linear (metres per radian), rows 3:6 angular
        (dimensionless). Column ``j`` is what joint ``j`` alone does to the tip."""
        tip, axes, anchors = self.forward(q)
        linear = torch.cross(axes, tip.unsqueeze(1) - anchors, dim=-1)
        return torch.cat([linear, axes], dim=-1).transpose(1, 2)

    def joint_state(self, robot_qpos: torch.Tensor) -> torch.Tensor:
        """The chain's own joints out of the observation's robot-qpos block."""
        return robot_qpos[:, self.joint_columns].float()


def null_space_direction(jac: torch.Tensor) -> torch.Tensor:
    """(B, n) unit vector spanning the chain's self-motion, or zeros where it collapses.

    A 7-hinge arm holding a 6-dof tool pose keeps exactly one degree of freedom: the elbow
    swinging about the shoulder-wrist axis while the tip stays put. This returns it in CLOSED
    FORM as the generalised cross product — component ``i`` is the signed minor of ``J`` with
    column ``i`` deleted — for which ``J n = 0`` holds EXACTLY, being the Laplace expansion of
    a square matrix with a repeated row.

    THE OBVIOUS CONSTRUCTION, ``(I - J^+ J) probe``, IS NOT EXACT AND THE ERROR IS NOT
    SMALL. `J^+` here is DAMPED, so that projector keeps an O(lambda^2) row-space residual;
    normalising a probe whose overlap with the true direction happens to be small then
    amplifies it, leaking tool motion (on the order of 1e-3) into a channel whose entire
    contract is that it moves the tool not at all.

    At a singularity the minors vanish together, which is the honest answer: an arm that has
    lost a Cartesian direction has no self-motion to spend either, and the zero switches the
    channel off for that tick instead of dividing by a vanishing norm.
    """
    B, m, n = jac.shape
    if n != m + 1:
        raise ValueError(
            f"the closed-form self-motion direction needs exactly one redundant degree of "
            f"freedom, but this chain has {n} joints against an {m}-dof tool frame.")
    keep = torch.stack([torch.cat([torch.arange(i, device=jac.device),
                                   torch.arange(i + 1, n, device=jac.device)])
                        for i in range(n)])                       # (n, m)
    # DOUBLE PRECISION for the minors alone. Each is a product of m entries, so a chain whose
    # columns are ~0.4 m/rad gives determinants around 1e-2 and a near-singular pose drives
    # them toward float32's 1e-7 epsilon, where the normalisation would divide noise by noise.
    wide = jac.double()
    minors = torch.linalg.det(wide[:, :, keep].permute(0, 2, 1, 3))              # (B, n)
    signs = torch.pow(-1.0, torch.arange(n, device=jac.device, dtype=minors.dtype))
    vector = signs * minors
    # RELATIVE threshold: an absolute one would mean different things on a 0.04 m cube scene
    # and a 0.16 m one, since every minor carries the chain's length scale to the m-th power.
    scale = wide.norm(dim=(1, 2)).pow(m).clamp_min(1e-30)
    norm = vector.norm(dim=-1)
    live = (norm > 1e-6 * scale).unsqueeze(-1)
    return torch.where(live, vector / norm.clamp_min(1e-30).unsqueeze(-1),
                       torch.zeros_like(vector)).to(jac.dtype)


def build_joint_chain(model, joint_names, tip_body: str, robot_qpos_index,
                      action_scale: float, device) -> JointChain:
    """Walk the compiled model from ``tip_body`` to the root and transcribe the chain.

    Walking the REAL body tree rather than assuming a flat list is what makes jointless
    carrier frames work, and folding each run of them into the next hinge's fixed transform
    is what keeps the batched loop seven iterations long instead of twelve.
    """
    wanted = list(joint_names)
    path, body = [], model.body(tip_body).id
    while body > 0:
        path.append(body)
        body = int(model.body_parentid[body])
    path.reverse()

    robot_qpos_index = list(np.asarray(robot_qpos_index).ravel())
    steps, seen, pending = [], [], np.eye(4)
    for b in path:
        name = model.body(b).name
        pending = pending @ homogeneous(quat_to_rotation(model.body_quat[b]), model.body_pos[b])
        n_joints = int(model.body_jntnum[b])
        if not n_joints:
            continue
        if n_joints != 1:
            raise SystemExit(
                f"body {name!r} carries {n_joints} joints; the chain transcription handles "
                f"one hinge per body.")
        j = int(model.body_jntadr[b])
        joint_name = model.joint(j).name
        if joint_name not in wanted:
            continue
        adr = int(model.jnt_qposadr[j])
        if adr not in robot_qpos_index:
            raise SystemExit(
                f"joint {joint_name!r} sits at qpos {adr}, which the observation's robot-qpos "
                f"block does not publish; the chain cannot be driven.")
        axis = np.asarray(model.jnt_axis[j], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        anchor = np.asarray(model.jnt_pos[j], dtype=np.float64)
        # R(t) = I + sin(t) K + (1-cos(t)) K^2 about `anchor`, as one 4x4 affine in the two
        # scalars — which is what lets every hinge in the chain be built in one batched op.
        k1 = skew(axis)
        k2 = k1 @ k1
        # T(t) = [[R(t), anchor - R(t) anchor], [0, 1]] with R = I + sin(t) K + (1-cos(t)) K^2,
        # so the two variable terms carry a ZERO bottom row -- the constant A holds the 1.
        term_b, term_c = np.zeros((4, 4)), np.zeros((4, 4))
        term_b[:3, :3], term_b[:3, 3] = k1, -k1 @ anchor
        term_c[:3, :3], term_c[:3, 3] = k2, -k2 @ anchor
        steps.append({
            "name": joint_name, "fixed": pending.copy(), "axis": axis, "anchor": anchor,
            "qpos0": float(model.qpos0[adr]),
            "A": np.eye(4), "B": term_b, "C": term_c,
        })
        seen.append(joint_name)
        pending = np.eye(4)

    if seen != wanted:
        raise SystemExit(
            f"the chain from the root to {tip_body!r} carries joints {seen}, but the "
            f"embodiment declares {wanted}. The declared order IS the action order, so a "
            f"mismatch would silently permute the Jacobian's columns.")
    columns = []
    for joint_name in wanted:
        j = model.joint(joint_name).id
        columns.append(robot_qpos_index.index(int(model.jnt_qposadr[j])))
    return JointChain(steps, pending, columns, action_scale, device)
