from dataclasses import dataclass

import torch


def quat_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (..., 4) (w, x, y, z) -> rotation matrices (..., 3, 3), row-major."""
    q = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
    ], dim=-2)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """(B, 4) wxyz -> (B, 3) axis-angle, shortest arc, small-angle safe."""
    q = torch.where(q[:, :1] < 0, -q, q)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    s = q[:, 1:].norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[:, :1])
    axis = q[:, 1:] / s.clamp_min(1e-12)
    return torch.where(s < 1e-6, 2.0 * q[:, 1:], angle * axis)


def rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """(B, 3) axis-angle -> (B, 4) wxyz, small-angle safe."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / angle.clamp_min(1e-12)
    xyz = torch.where(angle < 1e-6, 0.5 * rotvec, torch.sin(0.5 * angle) * axis)
    return torch.cat([torch.cos(0.5 * angle), xyz], dim=-1)


def truncate_xy(p: torch.Tensor, centre: torch.Tensor, radius: float) -> torch.Tensor:
    """(B, 3) hard truncation of `p` into the xy disc of radius R about `centre`.

    **z is passed through untouched** — the disc is a CYLINDER, not a ball: the floor is not
    translation-invariant, and shrinking a vertical offset would lie about the rung height
    of a stack. Exactly the identity for anything already inside, so it cannot perturb a
    configuration that already works.
    """
    d = p[:, :2] - centre[:, :2]
    n = d.norm(dim=-1, keepdim=True)
    xy = torch.where(n > radius, centre[:, :2] + d * (radius / n.clamp_min(1e-9)), p[:, :2])
    return torch.cat([xy, p[:, 2:3]], dim=1)


def carrot_project(obj: torch.Tensor, goal: torch.Tensor, hand: torch.Tensor,
                   radius: float):
    """Pure pursuit. Returns (object', goal').

        object' = hand    + u_hat * min(|x - hand|, R)      u_hat = (x - hand)/|x - hand|
        goal'   = object' + v_hat * min(|g - x|,    R)      v_hat = (g - x)/|g - x|

    Both directions are the true ones and only the magnitudes are truncated, so the perceived
    goal error is exactly `v_hat * min(|g - x|, R)` -- never rotated, never collapsing, and the
    identity whenever the true error is already inside R. `v_hat` is anchored on x, the true
    object position, not on `object'`, which would otherwise rotate the presented goal error
    whenever the object itself lies beyond R.
    """
    obj_p = truncate_xy(obj, hand, radius)
    d_xy = (goal - obj)[:, :2]
    n_xy = d_xy.norm(dim=-1, keepdim=True)
    goal_p = goal.clone()
    goal_p[:, :2] = obj_p[:, :2] + d_xy * (n_xy.clamp(max=radius) / n_xy.clamp_min(1e-9))
    return obj_p, goal_p


def band_shift_z(obj_z: torch.Tensor, carriage_z: torch.Tensor,
                 band_object, band_carriage) -> torch.Tensor:
    """The MINIMAL s_z putting BOTH presented quantities inside the skill's trained bands, and
    exactly 0 when they already are.

    Each entity contributes the interval of shifts satisfying it (`p - hi <= s <= p - lo`);
    the answer is the point of least magnitude in the intersection, i.e. 0 clamped into it.
    If the two intervals do not overlap the OBJECT's wins — it is the thing that has to be
    grasped and seated, and the carriage has the warp's offset ladder to spare.
    """
    o_lo, o_hi = band_object
    c_lo, c_hi = band_carriage
    lo = torch.maximum(obj_z - o_hi, carriage_z - c_hi)
    hi = torch.minimum(obj_z - o_lo, carriage_z - c_lo)
    empty = lo > hi
    lo = torch.where(empty, obj_z - o_hi, lo)
    hi = torch.where(empty, obj_z - o_lo, hi)
    return torch.clamp(torch.zeros_like(lo), min=lo, max=hi)


@dataclass(frozen=True)
class EmbodimentLayout:
    """Which action / robot-qpos channels mean what, for the robot the skill drives."""

    robot: str
    #: qpos/action indices commanding end-effector TRANSLATION (the leading prismatic run)
    ee_translation_qpos: tuple[int, ...]
    #: the jaw/finger dofs — the crop's `jaws` slot and the velocity partner of it
    gripper_qpos: tuple[int, ...]
    #: every other robot dof: an arm's joints, the gantry's three wrist hinges
    joint_qpos: tuple[int, ...]
    grip_action: int | None

    @property
    def has_grip(self) -> bool:
        return self.grip_action is not None

    @property
    def is_cartesian(self) -> bool:
        """True when the robot's leading joints ARE the EE's Cartesian position.

        False on any arm. It decides the SEMANTICS of the warp's perceived-EE groups, not
        whether they are live: on a Cartesian carriage the leading channels are shifted
        radially about the object, in metres; on an arm every robot_qpos channel is a joint
        angle and the same groups become a plain affine warp on the angles π₀ perceives, in
        radians, on the position ladder (0.1875 h per quantum read as radians). The
        ACTION-side warp follows the skill's real output channels either way."""
        return len(self.ee_translation_qpos) > 0


@dataclass
class SkillInputs:
    """The selected object and the robot, per scene, on device — the surface an input warp
    acts on. Everything a skill's observation is a function of, before the skill's own
    localisation and scaling."""

    robot_q: torch.Tensor        # (B, n_robot_qpos)
    robot_v: torch.Tensor        # (B, n_robot_qvel)
    servo: torch.Tensor          # (B, n_ctrl)  ctrl_ref - ctrl_pos
    obj_q: torch.Tensor          # (B, 7)
    obj_v: torch.Tensor          # (B, 6)
    obj_centre: torch.Tensor     # (B, 3)
    goal_centre: torch.Tensor    # (B, 3)
    goal_quat: torch.Tensor      # (B, 4)
    #: the UNWARPED worked-object centre. Radial position slopes are taken about this, and
    #: it is what makes a position slope well defined at all.
    anchor: torch.Tensor         # (B, 3)
    #: the UNWARPED end-effector position, in the frame obj_centre already uses
    hand: torch.Tensor           # (B, 3)
    #: the one warp term that reaches into the observation itself: it scales the goal-feature
    #: DELTA, which does not exist until the feature table has been applied.
    err_gain: torch.Tensor | None = None   # (B, 3)
    err_off: torch.Tensor | None = None    # (B, 3)


class FrozenSkill:
    """A sealed single-object policy plus the embodiment it drives.

    It owns three things no algorithm above it should know: how to build ITS OWN observation
    from a multi-object observation and an object index, how it localises a scene object
    into the span it was trained on, and how to run itself. Every lifter — a heuristic
    baseline, a fine-tuner, WAVE — goes through `present` / `act`, so a new robot is a new
    subclass here and nothing else changes.
    """

    #: the embodiment this subclass drives, as the scene sidecars name it
    embodiment: str = ""
    #: action index of the grip command, or None when the embodiment has none
    grip_action: int | None = None
    #: scene frame that IS the end effector, or None for a prismatic run
    ee_frame: str | None = None
    #: the ACTUATED joints from the robot's root to `ee_frame`, in ACTION-CHANNEL ORDER, for
    #: an embodiment whose action channels are joint targets. Declaring them lets the env
    #: transcribe the chain and hand back a `JointChain`, which is what a task-space output
    #: warp needs. A Cartesian carriage declares none: its leading channels ARE the tool
    #: frame, so there is nothing to map.
    chain_joints: tuple[str, ...] = ()
    #: how many robot dofs the embodiment has, when the joint chain under-counts them
    n_robot_dofs: int | None = None
    #: R of the xy disc the skill is presented inside; 0 = no truncation
    disc_radius: float = 0.0

    def __init__(self, layout: EmbodimentLayout, feature_points: torch.Tensor,
                 obs_dim: int, action_dim: int, eps_pos: float, device) -> None:
        self.layout = layout
        self.feature_points = feature_points     # (P, 3) on device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.eps_pos = eps_pos
        self.device = device
        #: the transcribed `JointChain`, attached by the env once the model is compiled, or
        #: None on an embodiment that declares no `chain_joints`.
        self.chain = None

    @property
    def n_feature_points(self) -> int:
        return self.feature_points.shape[0]

    def feature(self, centre: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """(B, 3(1+P)) the object feature the skill's scene declares: the centre followed by
        each rigidly attached marker at ``centre + R @ offset``. P = 0 makes the feature the
        centre itself, which is what a sphere pusher trained on."""
        if self.n_feature_points == 0:
            return centre
        offsets = self.feature_points[None] @ rotation.transpose(-1, -2)
        return torch.cat([centre, (centre[:, None, :] + offsets).reshape(centre.shape[0], -1)],
                         dim=1)

    def inputs(self, obs, idx: torch.Tensor, goal_override=None) -> SkillInputs:
        rows = torch.arange(obs.active_mask.shape[0], device=self.device)
        anchor = obs.obj_feat[rows, idx].float()
        goal_centre = (obs.goal_feat[rows, idx].float() if goal_override is None
                       else goal_override.float())
        goal_quat = torch.zeros(anchor.shape[0], 4, device=anchor.device, dtype=anchor.dtype)
        goal_quat[:, 0] = 1.0
        return SkillInputs(
            robot_q=obs.robot_qpos.float(),
            robot_v=obs.robot_qvel.float(),
            servo=obs.ctrl_ref.float() - obs.ctrl_pos.float(),
            obj_q=obs.obj_qpos[rows, idx].float(),
            obj_v=obs.obj_qvel[rows, idx].float(),
            obj_centre=anchor,
            goal_centre=goal_centre,
            goal_quat=goal_quat,
            anchor=anchor,
            hand=obs.ee_position.float(),
        )

    def localize(self, x: SkillInputs) -> SkillInputs:
        """Move the scene into the span the skill was trained on. The identity by default;
        a subclass that declares a reliable reach or a trained z band overrides it."""
        return x

    def observe(self, x: SkillInputs) -> torch.Tensor:
        raise NotImplementedError

    def present(self, obs, idx: torch.Tensor, warp=None, goal_override=None) -> torch.Tensor:
        """(B, obs_dim) — the skill's own observation for the selected objects.

        The warp acts on the raw struct, BEFORE localisation, so the warp keeps its
        world-frame meaning and localisation stays a pure change of frame. Localisation is
        part of the skill: it moves the scene into the span the skill was trained on, so every
        caller sees the skill at its own competence.

        This projection is LOSSY BY DESIGN: the skill stays blind to the other objects, and
        an algorithm that does see them compensates through the warp.
        """
        x = self.inputs(obs, idx, goal_override)
        if warp is not None:
            x = warp.apply(x, self.layout)
        return self.observe(self.localize(x))

    def act(self, skill_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def banner(self) -> str:
        raise NotImplementedError
