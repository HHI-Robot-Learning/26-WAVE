import math

import torch
from torch import nn
from torch.distributions import Categorical

from wave_rl.algorithms.crop import CROP_CORE, N_CLEARANCE, PAIR_DIM, CropBatch
from wave_rl.algorithms.wave.config import WarpHeadConfig
from wave_rl.skills.base import quat_to_rotvec, rotvec_to_quat
from wave_rl.skills.kinematics import null_space_direction

#: the Panda gripper's robot dofs (the gantry has 8: xyz + wrist x3 + two jaws)
MAX_ROBOT_DOFS = 9
#: the Panda gripper's actuators (the gantry has 7; the right jaw is a passive mirror)
MAX_ACTUATORS = 8
#: the Panda's q1..q7, the widest NON-GRIP action any declared embodiment has
MAX_ACTION_JOINTS = 7

#: (name, width), laid out slope-block THEN offset-block per part.
GROUP_SPECS = [
    ("goal_slope", 3), ("goal", 3),
    ("goal_rot_slope", 3), ("goal_rot", 3),
    ("obj_slope", 3), ("obj", 3),
    ("obj_rot_slope", 3), ("obj_rot", 3),
    ("obj_vel_gain", 3), ("obj_vel_off", 3),
    ("obj_angvel_gain", 3), ("obj_angvel_off", 3),
    ("ee_slope", MAX_ROBOT_DOFS), ("ee", MAX_ROBOT_DOFS),
    ("ee_vel_gain", MAX_ROBOT_DOFS), ("ee_vel_off", MAX_ROBOT_DOFS),
    ("err_gain", 3), ("err_off", 3),
    ("srv_gain", MAX_ACTUATORS), ("srv_off", MAX_ACTUATORS),
    ("act_gain", MAX_ACTION_JOINTS), ("act_t", MAX_ACTION_JOINTS),
    ("grip_gain", 1), ("grip", 1),
    # The one direction a per-channel affine on the joint command cannot name: motion along
    # the null space of J(q) -- the elbow, and not the tool. Seven independent per-joint
    # offsets span it in principle, but only as a configuration-dependent combination of all
    # seven that a factorised head never draws coherently, and a per-joint gain cannot scale
    # the skill's own null-space component at all. Pinned wherever the direction does not
    # exist (a carriage).
    ("swivel_gain", 1), ("swivel", 1),
]

#: applied as (1 + param)*x. Everything else is added.
SLOPE_GROUPS = {
    "goal_slope", "goal_rot_slope", "obj_slope", "obj_rot_slope", "obj_vel_gain",
    "obj_angvel_gain", "ee_slope", "ee_vel_gain", "err_gain", "srv_gain", "act_gain",
    "grip_gain", "swivel_gain",
}
#: the OUTPUT side of the interface — what `warp_side` splits on. Membership is by NAME so
#: adding a channel to the action side cannot desynchronize the ablation.
ACTION_GROUPS = {"act_gain", "act_t", "grip", "grip_gain", "swivel", "swivel_gain"}
#: offset groups measured in radians: they take u_rot, not the position quantum.
ROTATION_OFFSET_GROUPS = {"obj_rot", "goal_rot"}
#: offset groups measured in velocity units.
VELOCITY_OFFSET_GROUPS = {"obj_vel_off", "obj_angvel_off", "ee_vel_off"}

#: 112 — every channel of the skill's I/O plus the swivel, one slope and one offset each.
P_DIM = sum(w for _, w in GROUP_SPECS)

GROUP_SLICES = []
_offset = 0
for _name, _width in GROUP_SPECS:
    GROUP_SLICES.append((_offset, _offset + _width))
    _offset += _width


def split(applied: torch.Tensor) -> dict:
    """Committed warp params (B, P_DIM) -> named pieces for sigma and the action path."""
    return {name: applied[:, lo:hi]
            for (name, _), (lo, hi) in zip(GROUP_SPECS, GROUP_SLICES)}


def side_pinned_groups(side: str) -> tuple:
    """Group names `warp_side` pins to identity."""
    if side == "full_io":
        return ()
    if side == "input_only":
        return tuple(sorted(ACTION_GROUPS))
    if side == "output_only":
        return tuple(sorted(n for n, _ in GROUP_SPECS if n not in ACTION_GROUPS))
    raise ValueError(f"warp_side={side!r} is not full_io | input_only | output_only")


def component_pinned_groups(component: str) -> tuple:
    """Group names `warp_component` pins to identity. GROUP_SPECS alternates
    slope-then-offset, so the two halves are exactly SLOPE_GROUPS and its complement."""
    if component == "both":
        return ()
    if component == "slope_only":
        return tuple(sorted(n for n, _ in GROUP_SPECS if n not in SLOPE_GROUPS))
    if component == "offset_only":
        return tuple(sorted(SLOPE_GROUPS))
    raise ValueError(f"warp_component={component!r} is not both | slope_only | offset_only")


def surplus_pinned_groups(layout, action_dim: int, has_chain: bool) -> tuple:
    """`name@k` pins for the channels this embodiment does not have.

    Every group is sized to the widest declared robot; GROUP_SPECS and P_DIM stay untouched
    so checkpoints remain mutually loadable across embodiments. An arm has no prismatic run,
    so nothing is pinned for the `ee` warp's radial half — the warper must still be able to
    bend its own proprioception.
    """
    n_trans = len(layout.ee_translation_qpos)
    n_dofs = n_trans + len(layout.joint_qpos) + len(layout.gripper_qpos)
    n_motion = action_dim - (1 if layout.has_grip else 0)
    pins = []
    if not layout.has_grip:
        pins += ["grip", "grip_gain"]
    if not has_chain:
        pins += ["swivel", "swivel_gain"]
    pins += [f"ee@{n_dofs}", f"ee_slope@{n_dofs}",
             f"ee_vel_gain@{n_dofs}", f"ee_vel_off@{n_dofs}"]
    pins += [f"srv_gain@{action_dim}", f"srv_off@{action_dim}",
             f"act_gain@{n_motion}", f"act_t@{n_motion}"]
    return tuple(pins)


def warp_action(action: torch.Tensor, applied: torch.Tensor, layout, chain,
                robot_qpos: torch.Tensor) -> torch.Tensor:
    """Apply the output-side groups to the skill's raw action (B, A) as a per-channel affine:

        non-grip channel k:  a_k <- (1 + act_gain_k)*a_k + act_t_k
        grip:                g   <- (1 + grip_gain)*g    + grip
        on a chain:          a   <- a + swivel_gain * (a.z) z + swivel * z

    Every channel is warped alike, since the skill's raw action is dimensionless and already
    clamped to [-1, 1] — unit conversion to metres or radians happens downstream in
    `action_scale`.

    THE SWIVEL IS THE ONE MOTION A PER-CHANNEL AFFINE CANNOT ASK FOR. `act_gain`/`act_t` hold
    one column per joint, and a redundant arm has exactly as many joints as there are columns,
    so every word the warper can speak is already committed to a joint. `z`, the null
    direction of J(q), is the motion that is left over: on a 7-hinge arm holding a 6-dof pose
    it swings the elbow around the shoulder-wrist line, which is the only way to take the
    forearm out of the scene while the hand keeps the pose the skill asked for. `swivel_gain`
    scales the skill's OWN component along that direction, `swivel` adds one outright.

    THE SELF-MOTION COMPONENT IS TAKEN BY PROJECTING ONTO `z`, NOT AS `a - J^+ J a`: a
    damped `J^+` makes `J^+ J` an inexact projector, and that subtraction keeps a row-space
    residual which leaks tool motion into the one channel whose whole contract is that it
    moves the tool not at all. The projection carries no such term and needs no `J^+`.

    `chain.action_scale` CANCELS, which is why it does not appear. `z` is a unit vector in
    joint space, so projecting the command in radians (`s a`) onto it and mapping the
    resulting component back into action units divides by the same `s` that multiplied it.
    The projection is therefore taken on the raw action vector.

    THE SWIVEL IS CLAMPED PER CHANNEL with everything else: the affine underneath is
    per-joint and independent, and one saturated joint has no business shrinking the other
    six. A swivel that drives a joint into the rail loses the part of itself that does not
    fit.
    """
    p = split(applied)
    a = action.clone()
    n = a.shape[1]
    grip_idx = layout.grip_action
    n_joints = n
    if grip_idx is not None:
        assert grip_idx in (-1, n - 1), (
            f"{layout.robot}: grip_action={grip_idx} is not the last of {n} action "
            f"channels; the leading slice below assumes it is.")
        n_joints -= 1
    n_joints = min(n_joints, MAX_ACTION_JOINTS)
    a[:, :n_joints] = (a[:, :n_joints] * (1.0 + p["act_gain"][:, :n_joints])
                       + p["act_t"][:, :n_joints])
    if chain is not None:
        m = chain.n_joints
        z = null_space_direction(chain.twist_jacobian(chain.joint_state(robot_qpos)))
        # `action`, not `a`: the gain reads the skill's OWN redundancy resolution, which is a
        # property of the command it issued and not of what the per-joint affine did to it.
        own = (action[:, :m] * z).sum(dim=-1, keepdim=True) * z
        a[:, :m] = a[:, :m] + p["swivel_gain"] * own + p["swivel"] * z
    if grip_idx is not None:
        a[:, grip_idx] = (a[:, grip_idx] * (1.0 + p["grip_gain"].squeeze(1))
                          + p["grip"].squeeze(1))
    return a.clamp(-1.0, 1.0)


class InputWarp:
    """The input half of a committed warp, applied to a :class:`SkillInputs` struct.

    Every piece is the full affine ``x <- (1 + slope)*x + offset``. Position slopes are
    radial scalings about the worked object: a plain multiplicative scaling of a world
    coordinate depends on where the world's origin sits, a radial one does not. On a
    carriage the leading joints ARE the tool frame, so the perceived-EE warp is radial about
    the object there too.
    """

    def __init__(self, applied: torch.Tensor) -> None:
        self.p = split(applied)

    def apply(self, x, layout):
        w = self.p
        anchor = x.anchor

        obj_q = x.obj_q.clone()
        obj_q[:, :3] = anchor + (1.0 + w["obj_slope"]) * (obj_q[:, :3] - anchor) + w["obj"]
        rotvec = quat_to_rotvec(obj_q[:, 3:7])
        obj_q[:, 3:7] = rotvec_to_quat((1.0 + w["obj_rot_slope"]) * rotvec + w["obj_rot"])
        x.obj_q = obj_q

        obj_v = x.obj_v.clone()
        obj_v[:, :3] = obj_v[:, :3] * (1.0 + w["obj_vel_gain"]) + w["obj_vel_off"]
        if obj_v.shape[1] >= 6:
            obj_v[:, 3:6] = (obj_v[:, 3:6] * (1.0 + w["obj_angvel_gain"])
                             + w["obj_angvel_off"])
        x.obj_v = obj_v

        robot_q = x.robot_q.clone()
        n_q = min(robot_q.shape[1], w["ee"].shape[1])
        n_t = min(len(layout.ee_translation_qpos), n_q)
        if n_t:
            lead = robot_q[:, :n_t]
            robot_q[:, :n_t] = (anchor[:, :n_t]
                                + (1.0 + w["ee_slope"][:, :n_t]) * (lead - anchor[:, :n_t])
                                + w["ee"][:, :n_t])
        if n_q > n_t:
            rest = robot_q[:, n_t:n_q]
            robot_q[:, n_t:n_q] = (rest * (1.0 + w["ee_slope"][:, n_t:n_q])
                                   + w["ee"][:, n_t:n_q])
        x.robot_q = robot_q

        robot_v = x.robot_v.clone()
        n_v = min(robot_v.shape[1], w["ee_vel_gain"].shape[1])
        robot_v[:, :n_v] = (robot_v[:, :n_v] * (1.0 + w["ee_vel_gain"][:, :n_v])
                            + w["ee_vel_off"][:, :n_v])
        x.robot_v = robot_v

        servo = x.servo.clone()
        n_s = min(servo.shape[1], w["srv_gain"].shape[1])
        servo[:, :n_s] = (servo[:, :n_s] * (1.0 + w["srv_gain"][:, :n_s])
                          + w["srv_off"][:, :n_s])
        x.servo = servo

        x.obj_centre = anchor + (1.0 + w["obj_slope"]) * (x.obj_centre - anchor) + w["obj"]
        x.goal_centre = anchor + (1.0 + w["goal_slope"]) * (x.goal_centre - anchor) + w["goal"]
        rv = quat_to_rotvec(x.goal_quat)
        x.goal_quat = rotvec_to_quat((1.0 + w["goal_rot_slope"]) * rv + w["goal_rot"])
        x.err_gain = w["err_gain"]
        x.err_off = w["err_off"]
        return x


def _identity_first(slopes) -> list:
    """Order a slope support the way every downstream mechanism assumes: identity at index
    0, then the rest by increasing |slope|.

    Several mechanisms address bins by literal index rather than by value lookup: the init
    bias writes `p_zero_init` to bin 0, the magnitude ranking calls rank 0 the identity, and
    the hold slot reads the same convention. A support in plain `sorted()` order would put
    the identity bias on the most negative bin instead.
    """
    values = sorted({round(float(v), 9) for v in slopes})
    rest = sorted((v for v in values if v != 0.0), key=lambda v: (abs(v), v))
    return [0.0] + rest


def multiplier_ladder(multipliers) -> list:
    """Build a slope support from multipliers. `multipliers` gives the non-negative half
    (identity 1.0 and takeover 0.0 included); the negative half is mirrored, so the support is
    symmetric in m by construction.

    Specified in m, stored in slope. The warp applies `x <- m*x + offset` with `m = 1 + slope`,
    so a channel sets a multiplier, whose natural symmetry is multiplicative: the partner of
    "halve it" (m = 0.5) is "double it" (m = 2), not m = 1.5. It is stored as `slope = m - 1`
    because index 0 of the table must be the identity and the table is shared with the offset
    channels, whose identity is 0.
    """
    ms = sorted({round(float(v), 9) for v in multipliers}
                | {round(-float(v), 9) for v in multipliers})
    return _identity_first([round(m - 1.0, 9) for m in ms])


def signed_ladder(magnitudes) -> list:
    """[0, -m1, +m1, -m2, +m2, ...] — identity at index 0, signs interleaved per magnitude."""
    return [0.0] + [s * m for m in magnitudes for s in (-1.0, 1.0)]


#: the geometric floor, then even strides of 4 out to the ceiling.
_LADDER_STEPS = tuple([1, 2, 4, 8] + list(range(12, 65, 4)))


def offset_steps(max_quanta: int) -> tuple:
    """The offset ladder in quanta: 1-unit steps near the identity (the gentleness lever),
    then 4-unit strides out to the ceiling."""
    return tuple(k for k in _LADDER_STEPS if k <= max_quanta)


def build_bin_table(slope_bins: list, max_quanta: int, h: float, cfg: WarpHeadConfig):
    """(P_DIM, NB) bin-value table + validity mask. Bin 0 is the identity in every channel;
    shorter supports are padded and masked up to the widest.

    Each family's ceiling is a multiple of its own quantum (metres, radians, velocity units)
    rather than one shared ceiling. The action family's quantum is dimensionless and never
    rescaled by object size — unlike the position quantum, which scales with the object
    half-extent — because `act_t` and `grip` live in the clamped action space, where every
    joint type is dimensionally identical. `swivel` is in that family for the same reason: it
    is a displacement along a UNIT direction in joint space, added to the command in the same
    clamped action units every other joint word is written in, so it takes the action ladder
    unchanged and `swivel_gain`, a dimensionless slope, takes the slope ladder.
    """
    steps = offset_steps(max_quanta)
    position_bins = signed_ladder([k * cfg.quantum_position_h * h for k in steps])
    rotation_bins = signed_ladder([k * cfg.quantum_rotation for k in steps])
    velocity_bins = signed_ladder([k * cfg.quantum_velocity for k in steps])
    action_bins = signed_ladder([k * cfg.quantum_action for k in steps])

    n_bins = max(len(slope_bins), len(position_bins), len(rotation_bins),
                 len(velocity_bins), len(action_bins))
    table = torch.zeros(P_DIM, n_bins)
    mask = torch.zeros(P_DIM, n_bins, dtype=torch.bool)
    column = 0
    for name, width in GROUP_SPECS:
        if name in SLOPE_GROUPS:
            bins = slope_bins
        elif name in ACTION_GROUPS:
            bins = action_bins
        elif name in ROTATION_OFFSET_GROUPS:
            bins = rotation_bins
        elif name in VELOCITY_OFFSET_GROUPS:
            bins = velocity_bins
        else:
            bins = position_bins
        for _ in range(width):
            table[column, :len(bins)] = torch.as_tensor(bins, dtype=torch.float32)
            mask[column, :len(bins)] = True
            column += 1
    return table, mask


def magnitude_levels(table, mask, hold_idx: int = -1):
    """(P_DIM, NB) sign-symmetric magnitude RANK per bin: 0 = identity (and the hold slot),
    1 = the smallest non-zero magnitude, 2 = the next. The rank scheme behind shaped eps and
    the magnitude-decayed init."""
    levels = torch.zeros(table.shape[0], table.shape[1], dtype=torch.long)
    for c in range(table.shape[0]):
        ranked = sorted({round(abs(float(v)), 9)
                         for v, valid in zip(table[c], mask[c])
                         if bool(valid) and abs(float(v)) > 0.0})
        for b in range(table.shape[1]):
            if (not bool(mask[c, b]) or b == hold_idx
                    or abs(float(table[c, b])) == 0.0):
                continue
            levels[c, b] = 1 + ranked.index(round(abs(float(table[c, b])), 9))
    return levels


def decay_exponent_scale(table, mask, levels, base: float, ref_table, ref_mask):
    """(P_DIM, 1) per-channel multiplier on the magnitude-level exponent, solved so the shaped
    profile's expected |warp value| matches the reference ladder's.

    E(alpha) is monotonically decreasing (mass moves toward the smallest magnitude), so the
    bisection below is exact to tolerance.
    """
    ref_levels = magnitude_levels(ref_table, ref_mask)

    def expected(tab, msk, lvl, alpha):
        w = (base ** (-alpha * lvl.float())) * msk.float() * (lvl > 0).float()
        return (w * tab.abs()).sum(-1) / w.sum(-1).clamp_min(1e-12)

    target = expected(ref_table, ref_mask, ref_levels, 1.0)
    live = (levels > 0).any(-1)
    lo = torch.full((table.shape[0],), 0.02)
    hi = torch.full((table.shape[0],), 60.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_big = expected(table, mask, levels, mid.unsqueeze(-1)) > target
        lo = torch.where(too_big, mid, lo)
        hi = torch.where(too_big, hi, mid)
    alpha = (0.5 * (lo + hi)).unsqueeze(-1)
    smallest = torch.where(mask & (levels > 0), table.abs(),
                           torch.full_like(table, float("inf"))).amin(-1)
    unreachable = live & (smallest > target)
    if bool(unreachable.any()):
        print(f"-- decay calibration UNREACHABLE on {int(unreachable.sum())}/{len(live)} "
              f"channels (their smallest magnitude is above the reference expected warp): "
              f"those keep the raw base^-level profile.")
    alpha = torch.where(unreachable.unsqueeze(-1), torch.ones_like(alpha), alpha)
    return torch.where(live.unsqueeze(-1), alpha, torch.ones_like(alpha))


def drawability_clamp(exponent, mask, base: float, p_zero_init: float,
                      p_hold_init: float, hold_idx: int, floor: float = 4e-6):
    """Shrink each channel's decay exponent by the smallest factor that keeps every live bin
    drawable at init.

    Per channel, bisects a scale s in [0, 1] on the exponent until the smallest live bin's
    init probability reaches `floor`; s = 0 is the uniform profile so a solution always
    exists, and channels already clearing the floor keep s = 1 (a no-op).
    """
    out = exponent.clone()
    for c in range(exponent.shape[0]):
        rest = mask[c].clone()
        rest[0] = False
        has_hold = bool(mask[c][hold_idx])
        if has_hold:
            rest[hold_idx] = False
        if int(rest.sum()) <= 0:
            continue
        p_hold = p_hold_init if has_hold else 0.0
        budget = max(0.0, 1.0 - max(1e-6, p_zero_init - p_hold) - p_hold)

        def worst(s: float, channel=c, live=rest, budget=budget) -> float:
            w = (base ** (-s * exponent[channel])) * live.float()
            return budget * float(w[live].min()) / max(1e-12, float(w.sum()))

        if worst(1.0) >= floor:
            continue
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if worst(mid) >= floor:
                lo = mid
            else:
                hi = mid
        out[c] = exponent[c] * lo
    return out


def build_group_queries():
    """[(name, [global channel indices])] — one decoder query per (slope, offset) PAIR of
    GROUP_SPECS, slopes emitted before offsets."""
    bounds, offset = [], 0
    for _, width in GROUP_SPECS:
        bounds.append((offset, offset + width))
        offset += width
    queries = []
    for g in range(0, len(GROUP_SPECS), 2):
        s_lo, s_hi = bounds[g]
        o_lo, o_hi = bounds[g + 1]
        name = GROUP_SPECS[g][0].replace("_slope", "").replace("_gain", "")
        queries.append((name, list(range(s_lo, s_hi)) + list(range(o_lo, o_hi))))
    return queries


GROUP_QUERIES = build_group_queries()
N_QUERIES = len(GROUP_QUERIES)
MAX_QUERY_LEN = max(len(ch) for _, ch in GROUP_QUERIES)


def categorical(p: torch.Tensor) -> Categorical:
    """A mixture `p` as a distribution, built from log-probs floored at 1e-12. Built from
    `probs` instead, torch floors at float32 epsilon (1.2e-7), which the exploration mixture's
    rarest bins sit below, and their log-probs would lose their gradient."""
    return Categorical(logits=p.clamp_min(1e-12).log(), validate_args=False)


class PairEncoder(nn.Module):
    """phi: one pair row -> N_CLEARANCE bounded scores, reduced over the other objects by an
    elementwise MIN.

    The min is the whole design. It is the reduction the rest of this benchmark already uses
    everywhere it has to survive a change of object count (the selector's argmax, its pooled
    quantiles), and it is the only one whose value is a
    property of the scene rather than of how many things are in it: a mean and a softmax
    both dilute with N, so a head trained at six objects reads a different number at thirty
    for an unchanged neighbourhood.

    No channel is named. Nothing pins a column of the pair row to a channel, and every
    channel begins as a random projection of the pair row, so what the eight numbers mean is
    settled by the return and by nothing a person decided in advance. Each channel is its own
    ranking of the other objects and reports the bottom of that ranking; the channels are
    free to rank by different things.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(PAIR_DIM, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, N_CLEARANCE))

    def rows(self, pairs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, N, PAIR_DIM) x (B, N) -> (B, N, N_CLEARANCE), the per-pair scores, reading +1
        wherever there is no pair so an absent neighbour can never win the min."""
        row = torch.tanh(self.net(pairs))
        return torch.where(mask.unsqueeze(-1), row, torch.ones_like(row))

    def forward(self, pairs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, N, PAIR_DIM) x (B, N) -> (B, N_CLEARANCE). An empty row set reads +1 on every
        channel: nothing anywhere, maximum clearance."""
        return self.rows(pairs, mask).amin(dim=1)


class WarpPolicy(nn.Module):
    """crop -> one categorical per warp PARAMETER, plus a value head.

    Trunk: the crop's factored tokens -- the core, the clearances (a learned per-pair score
    min-reduced over the other objects), and the previous applied warp -- through a small
    TransformerEncoder.
    Head: one query token per channel GROUP through a TransformerDecoder over that memory,
    then an UNTIED per-group readout.

    Every live channel carries a hold bin: "carry the previous applied value", distinct from
    the identity and fed back through the previous-warp token to keep the process Markov.

    Draws stay conditionally independent given the crop, so the joint log-prob is exact in
    one parallel pass.
    """

    def __init__(self, h: float, disabled_groups: tuple, cfg: WarpHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.disabled_groups = tuple(disabled_groups)
        d = cfg.d_model
        self.d_model = d

        table, mask = build_bin_table(multiplier_ladder(cfg.damping_multipliers),
                                      cfg.offset_max_quanta, h, cfg)
        # `name` pins a whole group, `name@k` its channels from k on.
        whole, spans = set(), {}
        for entry in self.disabled_groups:
            name, _, spec = str(entry).partition("@")
            if not spec:
                whole.add(name)
                continue
            spans.setdefault(name, []).append(int(spec))
        column = 0
        for name, width in GROUP_SPECS:
            dead = set(range(width)) if ("all" in whole or name in whole) else {
                i for k in spans.get(name, ()) for i in range(k, width)}
            for i in range(width):
                if i in dead:
                    mask[column, 1:] = False
                column += 1

        live = mask[:, 1:].any(-1)
        table = torch.cat([table, torch.zeros(P_DIM, 1)], dim=1)
        mask = torch.cat([mask, live.unsqueeze(1)], dim=1)
        self.register_buffer("bin_table", table)
        self.register_buffer("bin_mask", mask)
        self.n_bins = table.shape[1]
        self.hold_idx = self.n_bins - 1

        levels = magnitude_levels(table, mask, self.hold_idx)
        self.register_buffer("bin_magnitude_level", levels, persistent=False)
        reference = build_bin_table(cfg.reference_slope_bins, cfg.reference_max_quanta, h, cfg)
        exponent = levels.float() * decay_exponent_scale(
            table, mask, levels, cfg.magnitude_decay_base, *reference)
        exponent = drawability_clamp(exponent, mask, cfg.magnitude_decay_base, cfg.p_zero_init,
                                     cfg.p_hold_init, self.hold_idx)
        self.register_buffer("bin_decay_exponent", exponent, persistent=False)
        weights = (cfg.magnitude_decay_base ** (-exponent)) * mask.float()
        self.register_buffer("epsilon_weights",
                             weights / weights.sum(-1, keepdim=True).clamp_min(1e-8),
                             persistent=False)
        #: per-channel scale normalising the previous-warp policy input to ~[-1, 1]
        self.register_buffer("prev_warp_scale", table.abs().amax(-1).clamp_min(1e-6),
                             persistent=False)

        self.register_buffer("norm_mean", torch.zeros(CROP_CORE))
        self.register_buffer("norm_var", torch.ones(CROP_CORE))
        self.register_buffer("norm_count", torch.tensor(1e-4))
        self.core_proj = nn.Linear(CROP_CORE, d)
        self.pair_encoder = PairEncoder(cfg.pair_hidden)
        self.clear_proj = nn.Linear(N_CLEARANCE, d)
        self.prev_warp_proj = nn.Linear(P_DIM, d)
        self.pos_emb = nn.Parameter(torch.zeros(3, d))

        encoder_layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.ff_mult * d, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_layers)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

        idx = torch.zeros(N_QUERIES, MAX_QUERY_LEN, dtype=torch.long)
        valid = torch.zeros(N_QUERIES, MAX_QUERY_LEN, dtype=torch.bool)
        for q, (_name, channels) in enumerate(GROUP_QUERIES):
            idx[q, :len(channels)] = torch.as_tensor(channels)
            valid[q, :len(channels)] = True
        self.register_buffer("query_channels", idx)
        self.register_buffer("query_valid", valid)
        flat_valid = valid.reshape(-1)
        self.register_buffer("scatter_source", torch.nonzero(flat_valid).squeeze(-1),
                             persistent=False)
        self.register_buffer("scatter_target", idx.reshape(-1)[flat_valid], persistent=False)

        self.query = nn.Parameter(torch.randn(N_QUERIES, d) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d, cfg.n_heads, cfg.ff_mult * d, dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.decoder_layers,
                                             norm=nn.LayerNorm(d))
        self.step_emb = nn.Parameter(torch.zeros(MAX_QUERY_LEN, d))

        self.w1 = nn.Parameter(torch.empty(N_QUERIES, d, d))
        self.b1 = nn.Parameter(torch.zeros(N_QUERIES, d))
        self.w2 = nn.Parameter(torch.empty(N_QUERIES, MAX_QUERY_LEN, d, self.n_bins))
        self.b2 = nn.Parameter(torch.zeros(N_QUERIES, MAX_QUERY_LEN, self.n_bins))
        with torch.no_grad():
            bound = 1.0 / math.sqrt(d)
            self.w1.uniform_(-bound, bound)
            self.b1.uniform_(-bound, bound)
            self.w2.uniform_(-bound, bound).mul_(0.01)
            self.b2.zero_()
        bias = self._identity_bias_table()
        self.channel_bias = nn.Parameter(bias)
        # Softmax first, then log: `_identity_bias_table` returns logits, so taking their
        # log directly gives log(log q), and rho would land outside the family the policy
        # can express. KL(pi || rho) must read ~0 at init, because there pi IS rho.
        with torch.no_grad():
            prior = torch.softmax(bias.masked_fill(~mask, -1e9), dim=-1) * mask.float()
        self.register_buffer("identity_prior_log", prior.clamp_min(1e-12).log(),
                             persistent=False)

    def _identity_bias_table(self) -> torch.Tensor:
        """(P_DIM, NB) logit bias putting `p_zero_init` on the identity bin of every live
        channel, `p_hold_init` on the hold slot, and the remainder shaped by the calibrated
        geometric decay over magnitude level. A pinned channel (one live bin) gets a flat zero
        bias — its mask already decides."""
        bias = torch.zeros(P_DIM, self.n_bins)
        for c in range(P_DIM):
            if int(self.bin_mask[c].sum()) - 1 <= 0:
                continue
            channel_mask = self.bin_mask[c]
            p_hold = self.cfg.p_hold_init
            p_zero = max(1e-6, self.cfg.p_zero_init - p_hold)
            rest = channel_mask.clone()
            rest[0] = False
            rest[self.hold_idx] = False
            base = self.cfg.magnitude_decay_base
            weights = (base ** (-self.bin_decay_exponent[c])) * rest.float()
            q = weights * (max(0.0, 1.0 - p_zero - p_hold) / max(1e-8, float(weights.sum())))
            q[0] = p_zero
            q[self.hold_idx] = p_hold
            bias[c] = q.clamp_min(1e-12).log() * channel_mask.float()
        return bias

    @torch.no_grad()
    def update_normalizer(self, crop: CropBatch) -> None:
        """Batched Welford update of the running core mean/var (trainer-only; frozen at eval).

        The core only. A pair row is bounded by construction -- compressed lengths, along-track
        fractions in [0, 1], flags -- and the clearances are a tanh, so there is nothing there
        whose scale a running estimate would have to discover."""
        core = crop.core.float()
        batch_mean = core.mean(dim=0)
        batch_var = core.var(dim=0, unbiased=False)
        batch_n = torch.tensor(float(core.shape[0]), device=core.device)
        n = self.norm_count + batch_n
        delta = batch_mean - self.norm_mean
        self.norm_mean += delta * (batch_n / n)
        self.norm_var.copy_((self.norm_var * self.norm_count + batch_var * batch_n
                             + delta.pow(2) * self.norm_count * batch_n / n) / n)
        self.norm_count.copy_(n)

    def clearance(self, pairs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, N, PAIR_DIM) x (B, N) -> (B, N_CLEARANCE): how much room the worked object has.

        Public because the selector reads the same eight numbers for every candidate, through
        this same encoder and detached, rather than growing a second opinion about clearance."""
        return self.pair_encoder(pairs.float(), mask)

    def _trunk(self, crop: CropBatch, prev_warp: torch.Tensor):
        x = (crop.core.float() - self.norm_mean) / (self.norm_var.sqrt() + 1e-6)
        clear = self.clearance(crop.pairs, crop.mask)
        tokens = [self.core_proj(x).unsqueeze(1), self.clear_proj(clear).unsqueeze(1),
                  self.prev_warp_proj(prev_warp.float() / self.prev_warp_scale).unsqueeze(1)]
        memory = self.encoder(torch.cat(tokens, dim=1) + self.pos_emb)
        return memory, memory.mean(dim=1)

    def _logits(self, memory: torch.Tensor) -> torch.Tensor:
        """(B, P_DIM, NB), invalid bins at -1e9."""
        B = memory.shape[0]
        q = self.query.unsqueeze(0).expand(B, -1, -1).contiguous()
        groups = self.decoder(q, memory)
        x = groups.unsqueeze(2) + self.step_emb.view(1, 1, -1, self.d_model)
        h = torch.nn.functional.gelu(
            torch.einsum("bqld,qde->bqle", x, self.w1) + self.b1.unsqueeze(0).unsqueeze(2))
        per_query = torch.einsum("bqle,qlek->bqlk", h, self.w2) + self.b2.unsqueeze(0)
        channels = self.query_channels
        per_query = per_query + self.channel_bias[channels].unsqueeze(0)
        per_query = per_query.masked_fill(~self.bin_mask[channels].unsqueeze(0), -1e9)
        return self._scatter(per_query)

    def _scatter(self, per_query: torch.Tensor) -> torch.Tensor:
        """(B, Q, L, NB) -> (B, P_DIM, NB) by channel index."""
        B = per_query.shape[0]
        flat = per_query.reshape(B, N_QUERIES * MAX_QUERY_LEN, per_query.shape[-1])
        out = torch.zeros(B, P_DIM, per_query.shape[-1], device=per_query.device,
                          dtype=per_query.dtype)
        out[:, self.scatter_target] = flat[:, self.scatter_source]
        return out

    def forward(self, crop: CropBatch, prev_warp: torch.Tensor):
        """-> (logits (B, P_DIM, NB), value (B,)).

        The value head reads the pooled trunk on purpose: V is a function of the state, and
        conditioning it on anything the action carries would make it an action-value and
        break the advantage baseline."""
        memory, pooled = self._trunk(crop, prev_warp)
        return self._logits(memory), self.value_head(pooled).squeeze(-1)

    def mixture_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """(1-eps)*softmax + eps*shaped — the permanent exploration quota.

        The single site both sampling and log-probs pass through, which makes the acting
        distribution and the scored distribution identical by construction so the PPO ratio
        stays exact. The eps mass is shaped by magnitude level rather than uniform, so every
        bin keeps permanent support — a bin that stops being sampled stops receiving
        gradient and can never recover.
        """
        eps, clamp = self.cfg.epsilon_exploration, self.cfg.logit_clamp
        p = torch.softmax(logits.clamp(-clamp, clamp), dim=-1)
        return (1.0 - eps) * p + eps * self.epsilon_weights

    def sample(self, crop: CropBatch, prev_warp: torch.Tensor):
        """-> (bin indices (B, P_DIM), joint log-prob (B,), value (B,), warp (B, P_DIM))."""
        memory, pooled = self._trunk(crop, prev_warp)
        logits = self._logits(memory)
        value = self.value_head(pooled).squeeze(-1)
        p = self.mixture_probs(logits)
        if not torch.isfinite(p).all():
            raise FloatingPointError(
                "the warp policy produced non-finite probabilities — the run has diverged; "
                "lower the learning rate.")
        dist = categorical(p)
        idx = dist.sample()
        return idx, dist.log_prob(idx).sum(-1), value, self.decode(idx, prev_warp)

    def log_prob_entropy(self, crop: CropBatch, idx: torch.Tensor, prev_warp: torch.Tensor):
        """-> (joint log-prob (B,), entropy (B,), value (B,), mixture probs (B,P_DIM,NB))."""
        memory, pooled = self._trunk(crop, prev_warp)
        logits = self._logits(memory)
        value = self.value_head(pooled).squeeze(-1)
        p = self.mixture_probs(logits)
        dist = categorical(p)
        return dist.log_prob(idx).sum(-1), dist.entropy().sum(-1), value, p

    def decode(self, idx: torch.Tensor, prev_warp: torch.Tensor) -> torch.Tensor:
        """bin indices -> the warp vector. The hold slot decodes to the carried previous value
        (zero after an episode boundary, so a hold can never leak across one)."""
        w = self.bin_table.unsqueeze(0).expand(idx.shape[0], -1, -1) \
            .gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return torch.where(idx == self.hold_idx, prev_warp.float(), w)

    @torch.no_grad()
    def greedy(self, crop: CropBatch, prev_warp: torch.Tensor):
        """-> (idx, warp, probs) — the deployed decision, taken through the same readout
        the policy trains. `forward` is not a shortcut for this: a head whose logits come
        off the group readout leaves any other Linear at its init forever."""
        memory, _ = self._trunk(crop, prev_warp)
        p = self.mixture_probs(self._logits(memory))
        idx = p.argmax(dim=-1)
        return idx, self.decode(idx, prev_warp), p

