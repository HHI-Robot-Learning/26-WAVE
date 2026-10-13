import math
from typing import NamedTuple

import torch

#: max action width across declared embodiments (the Panda's q1..q7 + grip); narrower
#: robots zero-pad, wider ones raise rather than truncate.
TOOL_ACT_DIM = 8
#: the robot's own velocity: 3 EE-translation dofs + the jaw dofs, the velocity partner of
#: the `jaws` qpos slot.
PLANT_VEL_DIM = 5
#: the robot's own joint state, positions then velocities: the non-EE-translation, non-jaw
#: dofs (the Panda's q1..q7, the gantry's wrist hinges), zero-padded to the widest.
#: Together with `jaws` and `plant velocity` this covers every actuated dof's position and
#: rate.
ROBOT_JOINT_DIM = 7

#: one row per (worked object, other object): where the other one is relative to the worked
#: object, to its goal and to the two corridors the skill will sweep, and where it has to go
#: itself. The scene enters every learned module through these rows and nothing else.
PAIR_DIM = 3 + 3 + 3 + 1 + 3 + 1 + 1 + 1
#: clearances per worked object: the min over the other objects of a learned per-pair score.
#: Nothing names them -- what the eight mean is settled by the return.
N_CLEARANCE = 8
#: the worked object and the robot -- the part of the crop that does not depend on how many
#: other objects there are.
CROP_CORE = (3 + 3 + 3 + 1 + 1 + 2 + TOOL_ACT_DIM + PLANT_VEL_DIM
             + 2 * ROBOT_JOINT_DIM)
#: the robot's COMPLETE proprioception as one block: jaw positions, plant velocity and
#: every non-jaw joint's position and rate. What the selector reads about the robot itself.
SELF_DIM = 2 + PLANT_VEL_DIM + 2 * ROBOT_JOINT_DIM


class CropBatch(NamedTuple):
    """What a decision reads: a fixed-width core, one row per other object, and the mask
    saying which of those rows exist. The rows are reduced to a fixed width INSIDE the
    policy, so nothing outside ever sees a width that moves with the object count."""

    core: torch.Tensor      # (B, CROP_CORE)
    pairs: torch.Tensor     # (B, N, PAIR_DIM)
    mask: torch.Tensor      # (B, N) bool


def carried_flag(obs, atom: torch.Tensor, supported_all: torch.Tensor,
                 h: float) -> torch.Tensor:
    """(B,) bool -- the worked object is carried: active, not resting on anything, and within
    grasp reach of the EE in xy. An object aloft near the jaws is held; one knocked mid-air far
    from the EE is neither carried nor supported.

    Measured xy-only against `obs.ee_position`, the grasp frame on every embodiment -- not
    against actuated joint coordinates: on a Cartesian carriage those happen to be x/y in
    metres, but on an arm they are joint angles in radians compared to an object position in
    metres, so the bit would fire on numerical coincidence.
    """
    rows = torch.arange(obs.active_mask.shape[0], device=obs.obj_feat.device)
    pos = obs.obj_feat.float()
    ee_xy = obs.ee_position[..., :2].float()
    near = torch.linalg.norm(pos[rows, atom, :2] - ee_xy, dim=-1) < 2.0 * h
    return (obs.active_mask > 0.5)[rows, atom] & ~supported_all[rows, atom] & near


def plant_velocity(obs, layout) -> torch.Tensor:
    """(B, PLANT_VEL_DIM) the robot's own velocity: the EE translation dofs plus the jaw
    dofs. Zero-padded so the width is constant across embodiments.

    Not a length and not compressed, for the same reason `obj_qvel` is not: a velocity is a
    rate, already in the plant's native units, and the crop's final clamp bounds it.
    """
    v = obs.robot_qvel.float()
    B = v.shape[0]
    translation = obs.ee_velocity.float()
    if translation.shape[1] < 3:
        translation = torch.cat(
            [translation, torch.zeros(B, 3 - translation.shape[1], device=v.device)], dim=1)
    aux = v[:, list(layout.gripper_qpos)]
    if aux.shape[1] < 2:
        aux = torch.cat([aux, torch.zeros(B, 2 - aux.shape[1], device=v.device)], dim=1)
    return torch.cat([translation, aux], dim=1)


def robot_joint_state(obs, layout) -> torch.Tensor:
    """(B, 2*ROBOT_JOINT_DIM) the robot's own joint positions then velocities, for the
    non-EE-translation, non-jaw dofs. Zero-padded so the width is constant across
    embodiments; raises rather than truncating, since a silently dropped joint is exactly
    the failure this block exists to end.

    Angles are left in radians and rates in rad/s: neither is a length, so neither is
    compressed or made object-relative, and the crop's final clamp bounds both.
    """
    idx = list(layout.joint_qpos)
    if len(idx) > ROBOT_JOINT_DIM:
        raise ValueError(
            f"{layout.robot} has {len(idx)} non-jaw joints but ROBOT_JOINT_DIM is "
            f"{ROBOT_JOINT_DIM}; raise it (and accept that CROP_CORE moves with it) rather "
            f"than truncate, or the crop quietly loses a joint.")
    B = obs.robot_qpos.shape[0]
    pad = ROBOT_JOINT_DIM - len(idx)
    q = obs.robot_qpos[:, idx].float()
    qd = obs.robot_qvel[:, idx].float()
    if pad:
        z = torch.zeros(B, pad, device=obs.robot_qpos.device)
        q = torch.cat([q, z], dim=1)
        qd = torch.cat([qd, z], dim=1)
    return torch.cat([q, qd], dim=1)


def pair_floor(pairs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, N, PAIR_DIM) x (B, N) -> (B, PAIR_DIM), the elementwise min of the pair rows over
    the other objects, zero where a scene has none. The scripted counterpart of a learned
    clearance: the same rows and the same reduction, with nothing to train — which is what
    lets a flat-state learner read the crop without a width that moves with the count."""
    floored = torch.where(mask.unsqueeze(-1), pairs, torch.full_like(pairs, float("inf")))
    return torch.where(mask.any(dim=1, keepdim=True), floored.amin(dim=1),
                       torch.zeros_like(pairs[:, 0]))


class Crop:
    """The one local view every learned module in the benchmark reads: the warper, the
    selector's candidate features, and the crop-conditioned TD7 baseline all share this
    definition.

    Every length is compressed as `log(1+m/L)/log(1+V/L)` rather than carried in raw metres,
    because raw metres carry the size of the environment (travel distances scale with arena
    diameter, neighbour spacing with arena density) and a policy could index on that instead.
    L is a local physical length (a multiple of the object's own half-extent); V is a fixed
    constant -- never the arena.
    """

    def __init__(self, h: float, disc_radius: float, layout, length_scale_h: float,
                 length_outer: float) -> None:
        self.h = float(h)
        #: L, the local length every crop magnitude is compressed about.
        self.length_scale = length_scale_h * self.h
        self.length_outer = length_outer
        self.disc_radius = float(disc_radius)
        self.layout = layout

    def banner(self) -> str:
        return (f"== CROP  core={CROP_CORE} pair={PAIR_DIM} clearances={N_CLEARANCE} | anchor="
                f"{'hand (disc R=%.3g)' % self.disc_radius if self.disc_radius else 'object'}"
                f" | length L={self.length_scale:.3f} m ({self.length_scale / self.h:.3g} h) "
                f"V={self.length_outer:.1f} (log-relative: relative resolution at every "
                f"scale; also the compact support of a pair row: beyond V of the object, the "
                f"goal and both corridors an object is masked out)")

    def self_state(self, obs) -> torch.Tensor:
        """(B, SELF_DIM) jaw positions, plant velocity and the arm block, zero-padded to the
        widest embodiment -- what the selector reads about the robot itself."""
        aux = obs.robot_qpos[:, list(self.layout.gripper_qpos)].float()
        if aux.shape[1] < 2:
            aux = torch.cat([aux, torch.zeros(aux.shape[0], 2 - aux.shape[1],
                                              device=aux.device)], dim=1)
        return torch.cat([aux, plant_velocity(obs, self.layout),
                          robot_joint_state(obs, self.layout)], dim=1)

    def compress(self, magnitude: torch.Tensor) -> torch.Tensor:
        """A magnitude -> a bounded magnitude, `log(1+m/L)/log(1+V/L)`.

        Direction is exact in this encoding; only the magnitude is compressed.
        """
        L = self.length_scale
        return torch.log1p(magnitude / L) / math.log1p(self.length_outer / L)

    def compress_vec(self, v: torch.Tensor) -> torch.Tensor:
        n = v.norm(dim=-1, keepdim=True)
        return v * (self.compress(n) / n.clamp_min(1e-9))

    def compress_signed(self, m: torch.Tensor) -> torch.Tensor:
        return torch.sign(m) * self.compress(m.abs())

    def pair_features(self, obs, sat: torch.Tensor, worked: torch.Tensor):
        """(pairs (B, M, N, PAIR_DIM), mask (B, M, N)) for the M worked indices in `worked`
        (B, M): every other active object, described relative to each worked object, to its
        goal, and to the two corridors the skill sweeps -- the fetch from the hand to the
        object, the carry from the object to its goal -- plus where that other object has to
        go itself.

            r           3   the other object relative to the worked one
            q           3   the other object relative to the worked one's goal
            w           3   the worked object relative to the hand
            dz          1   how much higher the other object stands
            u           3   the other object's goal relative to the worked one's goal
            sat         1   the other object is at its goal now
            |r|, |q|    2   the two plain distances, so a distance is never a detour
                            through three compressed components

        Every row is a difference of two positions and nothing else. Whatever a corridor,
        a projection or an approach column would have said is a function of these: the
        compression keeps direction exact and its magnitude map is monotone, so each vector
        inverts, and then the carry segment is `r - q` and the hand's line to the other
        object is `r + w`. Which of those are worth computing is left to phi, so no belief
        about how the frozen skill moves is written into the input.

        An object farther than V from the worked object, from its goal and from both the
        lines between them is masked out. That compact support is what makes the reduction
        over these rows exact under extension: an object added out there cannot change
        anything any consumer computes, at any object count. The two line distances are a
        claim about what provably CANNOT matter, so they are worth being generous about;
        they bound the support and are never handed to phi as a feature.
        """
        pos = obs.obj_feat.float()
        goal = obs.goal_feat.float()
        active = obs.active_mask > 0.5
        N = pos.shape[1]
        M = worked.shape[1]
        idx = worked.unsqueeze(-1).expand(-1, -1, 3)
        p_i = torch.gather(pos, 1, idx)[:, :, None, :]
        g_i = torch.gather(goal, 1, idx)[:, :, None, :]
        e = obs.ee_position.float()[:, None, None, :]
        p_j = pos[:, None, :, :]
        g_j = goal[:, None, :, :]

        r = p_j - p_i
        q = p_j - g_i
        # SUPPORT ONLY -- d_c and d_f bound which rows can matter, and go no further.
        s = g_i - p_i
        t_c = ((r * s).sum(-1) / s.pow(2).sum(-1).clamp_min(1e-9)).clamp(0.0, 1.0)
        d_c = torch.linalg.norm(r - t_c.unsqueeze(-1) * s, dim=-1)
        s_f = (p_i - e)[..., :2]
        r_f = (p_j - e)[..., :2]
        t_f = ((r_f * s_f).sum(-1) / s_f.pow(2).sum(-1).clamp_min(1e-9)).clamp(0.0, 1.0)
        d_f = torch.linalg.norm(r_f - t_f.unsqueeze(-1) * s_f, dim=-1)
        r_n = torch.linalg.norm(r, dim=-1)
        q_n = torch.linalg.norm(q, dim=-1)

        rho = torch.minimum(torch.minimum(r_n, q_n), torch.minimum(d_c, d_f))
        same = torch.arange(N, device=pos.device)[None, None, :] == worked[:, :, None]
        mask = active[:, None, :] & ~same & (rho < self.length_outer)

        feats = torch.cat([
            self.compress_vec(r), self.compress_vec(q),
            self.compress_vec(p_i - e).expand(-1, -1, N, -1),
            self.compress_signed(r[..., 2]).unsqueeze(-1),
            self.compress_vec(g_j - g_i),
            sat.float()[:, None, :, None].expand(-1, M, -1, -1),
            self.compress(r_n).unsqueeze(-1), self.compress(q_n).unsqueeze(-1),
        ], dim=-1).clamp(-5.0, 5.0)
        return torch.where(mask.unsqueeze(-1), feats, torch.zeros_like(feats)), mask

    def build(self, obs, atom: torch.Tensor, sat: torch.Tensor,
              carried: torch.Tensor, supported: torch.Tensor,
              tool_action: torch.Tensor | None) -> CropBatch:
        """The crop at the worked object. `core` is the worked object and the robot:

            x* - anchor    3    the worked object relative to the anchor
            v*             3    its linear velocity (a rate -- never compressed)
            g* - anchor    3    its declared goal
            carried        1
            supported      1
            jaws           2    the jaw dofs (plant state)
            tool action    8    the skill's action at the unwarped input, zero-padded (zero
                                when the crop does not observe the tool)
            plant velocity 5    the robot's own qdot, zero-padded
            arm state     14    every non-jaw joint's position then velocity

        and `pairs`/`mask` are the rest of the scene as `pair_features` describes it: one
        row per other object, reduced to a fixed width inside whoever reads it.
        """
        B = obs.obj_feat.shape[0]
        rows = torch.arange(B, device=obs.obj_feat.device)
        pos = obs.obj_feat.float()
        worked = pos[rows, atom]
        ee = obs.ee_position.float()
        goal = obs.goal_feat[rows, atom].float()
        velocity = obs.obj_qvel[rows, atom, :3].float()

        aux = obs.robot_qpos[:, list(self.layout.gripper_qpos)].float()
        if aux.shape[1] < 2:
            aux = torch.cat([aux, torch.zeros(B, 2 - aux.shape[1], device=aux.device)], dim=1)

        anchor = ee if self.disc_radius else worked

        if tool_action is None:
            tool = torch.zeros(B, TOOL_ACT_DIM, device=pos.device)
        elif tool_action.shape[1] > TOOL_ACT_DIM:
            raise ValueError(
                f"the frozen skill commands {tool_action.shape[1]} action channels but "
                f"TOOL_ACT_DIM is {TOOL_ACT_DIM}; raise it (and accept that CROP_CORE moves "
                f"with it) rather than truncate, or this block quietly loses a channel.")
        else:
            tool = torch.cat([tool_action.float(),
                              torch.zeros(B, TOOL_ACT_DIM - tool_action.shape[1],
                                          device=pos.device)], dim=1)

        core = torch.cat([
            self.compress_vec(worked - anchor if self.disc_radius else ee - worked),
            velocity,
            self.compress_vec(goal - anchor),
            carried.float().unsqueeze(1), supported.float().unsqueeze(1),
            aux, tool, plant_velocity(obs, self.layout), robot_joint_state(obs, self.layout),
        ], dim=1).clamp(-5.0, 5.0)
        pairs, mask = self.pair_features(obs, sat, atom.unsqueeze(1))
        return CropBatch(core, pairs[:, 0], mask[:, 0])
