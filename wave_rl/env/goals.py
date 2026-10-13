import functools

import numpy as np
import torch

from wave_rl.config import GoalConfig
from wave_rl.sim.scene import ObjectGeometry

def supported_by_set(rel_xy: torch.Tensor, dz: torch.Tensor, h: float,
                     valid: torch.Tensor, cfg: GoalConfig) -> torch.Tensor:
    """(...,) bool, the geometry of "object j holds object k up". A single support must sit
    under the resting centre; two supports may instead straddle it. `rel_xy` (..., J, 2) =
    candidate support centre - resting centre (or slot); `dz` (..., J) = resting z -
    candidate z; `valid` (..., J) masks the candidates that may count (active, not self)."""
    level = ((dz - 2.0 * h).abs() < cfg.support_level_dz_h * h) & valid
    overlap = (rel_xy.abs() < cfg.overlap_support_xy_h * h).all(-1) & level
    single = (torch.linalg.norm(rel_xy, dim=-1) < cfg.single_support_xy_h * h) & level
    r = torch.where(overlap.unsqueeze(-1), rel_xy, torch.zeros_like(rel_xy))
    opposite = torch.einsum("...jc,...kc->...jk", r, r) < 0.0
    straddle = opposite & overlap.unsqueeze(-1) & overlap.unsqueeze(-2)
    return single.any(-1) | straddle.flatten(-2).any(-1)


class StructureGoal:
    """The objective of a multi-object task, evaluated on the live world state: targets that
    form a SUPPORTED STRUCTURE — slots at fixed offsets from an anchor, a support DAG between
    them, and satisfaction gated bottom-up.

    It answers exactly what the env and the algorithms need and owns nothing else:

      - :meth:`satisfied` — the per-object boolean the env's success machinery consumes.
        Per-object is the interface, not a claim that objectives decompose: a relational
        objective computes its mask globally and broadcasts it.
      - :meth:`supported` — the universal statics predicate: the object rests on the floor or
        on another object directly beneath it. No task structure in it at all.
      - :meth:`place_goals` / :meth:`compile_goals` — where the targets go, and how they are
        recomputed from the world as it actually comes to rest.

    Not the goal's concern: whether reached objects get pinned, when the episode terminates
    (the env's ``success_mode``), and which robot is present.

    Everything that follows from "these objects hold each other up" lives here exactly once —
    the placement predicate, the DAG gate, realized-support retargeting and the statics
    conjunct — so a concrete structure writes only its LAYOUT: :meth:`structure_size`,
    :meth:`slot_offsets`, :meth:`level_of`, :meth:`supports_of` and :attr:`xy_eps`. Add an
    objective by subclassing and registering it in :func:`make_task_goal`.
    """

    def __init__(self, object: ObjectGeometry, eps_pos: float, eps_vel: float,
                 stack_dz: float, cfg: GoalConfig) -> None:
        self.cfg = cfg
        #: h, the object half-extent every length in this objective is a multiple of
        self.h = object.half_extent
        #: object-centre height of a flat rest on the support surface
        self.rest_z = object.rest_z
        self.eps_pos = eps_pos
        #: rest-speed tolerance; None = position-only
        self.eps_vel = eps_vel if eps_vel > 0.0 else None
        #: vertical rung spacing between courses (m)
        self.stack_dz = stack_dz
        self._layout_cache = {}

    def structure_size(self) -> int:
        """Objects per independent structure; 0 = a single structure that grows with N."""
        raise NotImplementedError

    def slot_offsets(self, n: int) -> np.ndarray:
        """(n, 3) slot offsets from the owning structure's anchor, in metres."""
        raise NotImplementedError

    def level_of(self, n: int) -> np.ndarray:
        """(n,) support depth; 0 = rests on the floor."""
        raise NotImplementedError

    def supports_of(self, n: int) -> np.ndarray:
        """(n, K) support indices, -1 = the floor. K may be 0 (a flat arrangement)."""
        raise NotImplementedError

    @property
    def xy_eps(self) -> float:
        """Lateral acceptance radius."""
        raise NotImplementedError

    def place_goals(self, rest_obj: torch.Tensor, arange_n: torch.Tensor,
                    active_mask: torch.Tensor) -> torch.Tensor:
        """Map a draw of resting per-object poses (B, N, 7) to this objective's TARGET poses.
        `active_mask` (B, N) is this episode's object count per scene, which a structure whose
        LAYOUT depends on the count must read (see :meth:`_layout_tables`).

        This is the goal's OWN sampling seam: the env only supplies a resting draw and asks
        where the targets go, so it never needs to know what the objective looks like."""
        raise NotImplementedError

    def fits(self, n: int) -> bool:
        """Whether an n-object layout fits the reachable part of the arena at all."""
        return True

    def supported(self, feats: torch.Tensor, active_mask: torch.Tensor,
                  floor_z: torch.Tensor) -> torch.Tensor:
        """(B, N) bool — the object rests on something: the floor (its centre within
        `floor_tolerance_h` of ``floor_z``) or another active object's top directly beneath it.

        ``floor_z`` (B,) is the caller's floor reference. It must be frozen per episode: a live
        min-z un-grounds every floor object the moment one is pressed into the floor mid-grasp.
        """
        n = feats.shape[1]
        on_floor = feats[..., 2] <= floor_z[:, None] + self.cfg.floor_tolerance_h * self.h
        rel_xy = feats[:, None, :, :2] - feats[:, :, None, :2]
        dz = feats[:, :, None, 2] - feats[:, None, :, 2]
        eye = torch.eye(n, dtype=torch.bool, device=feats.device)[None]
        under = supported_by_set(rel_xy, dz, self.h, ~eye & active_mask[:, None, :], self.cfg)
        return (on_floor | under) & active_mask

    def _layout_tables(self, n: int, device, dtype):
        """(offsets (n,n,3), level (n,n), supports (n,n,K)) — ROW c-1 IS THE LAYOUT OF A
        c-OBJECT EPISODE, its first c rows that episode's own layout and the rest carried
        over from the capacity-n layout, so a structure whose layout does not depend on the
        count reads back bit-identically to a single `slot_offsets(n)`.

        THE LAYOUT OF A GROWING STRUCTURE IS A FUNCTION OF THE EPISODE'S COUNT, NOT OF THE
        SCENE'S OBJECT CAPACITY. Taking it from the capacity — `slot_offsets(rest_obj.shape[1])`
        — would make every count in a run share one lattice, and then TRAINING AND EVALUATION
        WOULD DISAGREE ABOUT WHAT THE TASK IS, because they build scenes of different capacity
        for the same count (the sweep rolls every count on one 30-object scene).
        """
        ck = (n, str(device), str(dtype))
        if ck in self._layout_cache:
            return self._layout_cache[ck]
        full_off = np.asarray(self.slot_offsets(n), dtype=np.float64)
        full_lev = np.asarray(self.level_of(n), dtype=np.int64)
        full_sup = np.asarray(self.supports_of(n), dtype=np.int64).reshape(n, -1)
        per = [(np.asarray(self.slot_offsets(c), dtype=np.float64),
                np.asarray(self.level_of(c), dtype=np.int64),
                np.asarray(self.supports_of(c), dtype=np.int64).reshape(c, -1))
               for c in range(1, n + 1)]
        k = max([full_sup.shape[1]] + [sup.shape[1] for _, _, sup in per])

        def widen(sup):
            out = np.full((sup.shape[0], k), -1, dtype=np.int64)
            out[:, :sup.shape[1]] = sup
            return out

        tail_off, tail_lev, tail_sup = full_off, full_lev, widen(full_sup)
        offs = np.stack([np.concatenate([o, tail_off[c:]]) for c, (o, _, _) in enumerate(per, 1)])
        levs = np.stack([np.concatenate([lv, tail_lev[c:]]) for c, (_, lv, _) in enumerate(per, 1)])
        sups = np.stack([np.concatenate([widen(sp), tail_sup[c:]])
                         for c, (_, _, sp) in enumerate(per, 1)])
        out = (torch.as_tensor(offs, device=device, dtype=dtype),
               torch.as_tensor(levs, device=device, dtype=torch.long),
               torch.as_tensor(sups, device=device, dtype=torch.long))
        self._max_level_n = int(levs.max(initial=0))
        self._layout_cache[ck] = out
        return out

    def scene_layout(self, active_mask: torch.Tensor, dtype=None):
        """(offsets (B,n,3), level (B,n), supports (B,n,K)) — each scene's own layout, chosen
        by ITS OWN object count."""
        n = active_mask.shape[1]
        dt = torch.get_default_dtype() if dtype is None else dtype
        off, lev, sup = self._layout_tables(n, active_mask.device, dt)
        pick = active_mask.sum(dim=1).clamp(1, n) - 1
        return off[pick], lev[pick], sup[pick]

    def anchor_of(self, n: int) -> np.ndarray:
        """(n,) index of the object whose resting draw anchors each object's structure — the
        first index of its contiguous block. Independent of the count: a structure's block is
        the same wherever the structure sits."""
        s = self.structure_size()
        if s <= 0:
            return np.zeros(n, dtype=int)
        return (np.arange(n) // s) * s

    @functools.cached_property
    def max_fanin(self) -> int:
        """Width K of the support table — 0 for a floor arrangement, 1 for a column, 2 for a
        pyramid course. A constant of the goal spec, memoized because it is read every tick."""
        return self.supports_of(max(self.structure_size(), 2)).shape[1]

    def _placed(self, feats, goals_feat, active_mask):
        """(B, N) bool: the object's centre is over its slot footprint (loose xy) AND at the
        right height. Position only — the core of the success predicate; velocity is layered
        on only where success needs it."""
        xy = torch.linalg.norm(feats[..., :2] - goals_feat[..., :2], dim=2)
        dz = (feats[..., 2] - goals_feat[..., 2]).abs()
        return (xy <= self.xy_eps) & (dz <= self.eps_pos) & active_mask

    def _floor_z(self, goals_feat, active_mask):
        """(B,) the structure's own floor reference: the lowest level-0 slot height. Read from
        the goal layout rather than from the live world, so an object pressed into the floor
        mid-grasp cannot un-ground everything else."""
        _, lev, _ = self.scene_layout(active_mask, goals_feat.dtype)
        g_z = goals_feat[..., 2].clone()
        g_z[~((lev == 0) & active_mask)] = float("inf")
        return g_z.min(dim=1).values

    def _gate_dag(self, in_place: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """(B, N) AND each object with ALL of its supports, level by level ascending (so a
        support is already gated by the time its dependent is). Independent structures never
        reference each other, so they stay independent subgraphs.

        One pass per course over the whole batch: the support table is per scene, so which
        rows sit on which course differs between scenes of different counts."""
        if self.max_fanin == 0:
            return in_place
        _, lev, sup = self.scene_layout(active_mask)
        b, n, k = sup.shape
        valid = sup >= 0
        idx = sup.clamp(min=0).reshape(b, n * k)
        out = in_place.clone()
        for level in range(1, self._max_level_n + 1):
            held = (out.gather(1, idx).reshape(b, n, k) | ~valid).all(dim=2)
            out = torch.where(lev == level, out & held, out)
        return out

    def satisfied(self, feats: torch.Tensor, goals_feat: torch.Tensor,
                  speeds: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """(B, N) bool: is each object satisfied at this tick? Device tensors in and out; this
        is the GPU hot path, so it stays pure tensor ops. Inactive slots are always False."""
        in_place = self._placed(feats, goals_feat, active_mask)
        if self.eps_vel is not None:
            in_place = in_place & (speeds <= self.eps_vel)
        in_place = in_place & self.supported(feats, active_mask,
                                             self._floor_z(goals_feat, active_mask))
        return self._gate_dag(in_place, active_mask) & active_mask

    def compile_goals(self, nominal, feats, status, active_mask):
        """Map the NOMINAL targets (B, N, 3) to the COMMANDED ones given the realized world. The
        env calls it every step for every algorithm, so the compilation is shared.

        A dependent's commanded slot is recomputed from where its supports ACTUALLY came to
        rest — the CENTROID of the realized support centres, plus ``stack_dz`` in z. For a
        single support that is "directly on top of it"; for a pyramid course resting in the
        groove between two bricks the centroid IS the nestled position. Applied only once
        every support of that object satisfies its own goal; otherwise the nominal slot
        stands.

        Without it a support placed 1.5 cm off — inside tolerance — leaves the dependent's
        nominal slot floating misaligned and the structure falls with every component working,
        which measures tolerance compounding rather than the thing under test."""
        if self.max_fanin == 0:
            return nominal
        _, _, sup = self.scene_layout(active_mask, nominal.dtype)
        b, n, k = sup.shape
        c = feats.shape[-1]
        valid = sup >= 0
        idx = sup.clamp(min=0).reshape(b, n * k)
        cen = feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, c)).reshape(b, n, k, c)
        w = valid.to(cen.dtype).unsqueeze(-1)
        slot = (cen * w).sum(dim=2) / w.sum(dim=2).clamp(min=1.0)
        slot = slot + torch.as_tensor([0.0, 0.0, self.stack_dz],
                                      device=slot.device, dtype=slot.dtype)
        held = ((status.gather(1, idx).reshape(b, n, k) | ~valid).all(dim=2)
                & valid.any(dim=2))
        return torch.where(held.unsqueeze(-1), slot, nominal)


class TowersGoal(StructureGoal):
    """Multi-pile stacking: ``N`` objects form ``ceil(N / tower_height)`` INDEPENDENT towers of
    height ``tower_height``. Count scales via the number of piles, never via tower height —
    height is a precision/physics axis that would confound the count-OOD curve if scaled.
    Pile ``p`` owns the contiguous index block ``[p*h, (p+1)*h)`` and is anchored at its own
    base object's drawn resting pose.

    ``tower_height = 0`` means ONE tower of every object, which is the single-pile task.

    Piles stand metres apart, so lateral placement is relaxed to "anywhere over the support"
    rather than the strict fraction of a pitch a contact structure needs.
    """

    def __init__(self, tower_height: int, **kw):
        super().__init__(**kw)
        assert tower_height >= 0, f"tower_height must be >= 0, got {tower_height}"
        self.tower_height = tower_height

    def _height(self, n: int) -> int:
        return self.tower_height if self.tower_height > 0 else n

    def structure_size(self) -> int:
        return self.tower_height

    def level_of(self, n: int) -> np.ndarray:
        return np.arange(n) % self._height(n)

    def supports_of(self, n: int) -> np.ndarray:
        sup = (np.arange(n) - 1).reshape(n, 1)
        sup[self.level_of(n) == 0] = -1
        return sup

    def slot_offsets(self, n: int) -> np.ndarray:
        off = np.zeros((n, 3))
        off[:, 2] = self.level_of(n) * self.stack_dz
        return off

    @property
    def xy_eps(self) -> float:
        """The centre may sit anywhere over the lower object's top face."""
        return 0.5 * self.stack_dz

    def place_goals(self, rest_obj: torch.Tensor, arange_n: torch.Tensor,
                    active_mask: torch.Tensor) -> torch.Tensor:
        _, lev, _ = self.scene_layout(active_mask, rest_obj.dtype)
        base = (arange_n[None, :] - lev).unsqueeze(-1).expand(-1, -1, 3)
        tower = torch.zeros_like(rest_obj)
        tower[..., :3] = rest_obj[..., :3].gather(1, base)
        tower[..., 2] = self.rest_z + lev.to(rest_obj.dtype) * self.stack_dz
        tower[..., 3] = 1.0
        return tower


class LatticeGoal(StructureGoal):
    """Base for objectives whose targets form a **contact structure**: a lattice of slots whose
    neighbours TOUCH, instead of the independent, metres-apart piles :class:`TowersGoal` builds.
    The lattice pitch is the rung spacing ``stack_dz``, so a lattice of cubes is flush.

    Every subclass writes only its LAYOUT — :meth:`slot_offsets`, :meth:`level_of`,
    :meth:`supports_of`, :meth:`structure_size` — and inherits placement, gating, retargeting
    and statics from :class:`StructureGoal`. What this class adds on top is
    everything that follows from slots TOUCHING: a lateral tolerance that is a strict fraction
    of the pitch (so the acceptance regions of adjacent slots never intersect, or a cube
    between two slots would satisfy one of them and the structure would certify without being
    flush), rigid separation of independent structures, and a rigid fit into the reachable
    part of the arena.
    """

    def __init__(self, xy_frac: float, placeable_xy: float, **kw):
        super().__init__(**kw)
        self.pitch = self.stack_dz
        self.xy_frac = xy_frac
        assert 0.0 < self.xy_frac < 0.5, (
            f"lattice_xy_frac must be in (0, 0.5) so neighbouring slots stay disjoint; "
            f"got {self.xy_frac}. At >= 0.5 an object sitting between two slots satisfies one "
            f"of them and the structure certifies without being flush.")
        #: Reachability bound on a goal slot's |x| and |y| (m): the largest |xy| the scene's
        #: config pool ever rests an object at. Without it a lattice anchored at object 0's
        #: drawn rest pose can place far slots inside a containment wall, where no object can
        #: rest.
        self.placeable_xy = placeable_xy
        self._anchor_cache = {}

    @property
    def xy_eps(self) -> float:
        """Lateral acceptance radius — a strict fraction of the PITCH, so the acceptance
        regions of adjacent slots never intersect (see the class docstring)."""
        return self.xy_frac * self.pitch

    def _separate(self, slots_xy: torch.Tensor, n: int, active_mask: torch.Tensor) -> torch.Tensor:
        """RIGID per-structure xy shifts that make the structures of a MULTI-structure lattice
        disjoint. Returns ``(B, N, 2)`` shifts (zeros for a single structure).

        Each structure is anchored at its own object's drawn rest pose and nothing else keeps
        two anchors apart, so two structures can be drawn on top of each other: an episode
        unwinnable by construction. Structures keep their own draw and are only pushed apart
        when their footprints collide: the footprint is a structure's slot bounding box grown
        by half a pitch (so slot centres of different structures end up at least one pitch
        apart) plus a 2 cm margin; a colliding pair moves apart symmetrically along the axis
        that needs the smaller displacement, and every pass folds a structure that left the
        arena back inside |xy| <= placeable_xy, so separation and reachability converge
        together.
        """
        s = self.structure_size()
        B, dev, dt = slots_xy.shape[0], slots_xy.device, slots_xy.dtype
        if s <= 0 or n <= s:
            return torch.zeros(B, n, 2, device=dev, dtype=dt)
        k = (n + s - 1) // s
        sid = torch.arange(n, device=dev) // s
        idx = sid.view(1, n, 1).expand(B, n, 2)
        act = active_mask.unsqueeze(-1)
        lo = torch.full((B, k, 2), float("inf"), device=dev, dtype=dt).scatter_reduce(
            1, idx, torch.where(act, slots_xy, float("inf")), "amin", include_self=True)
        hi = torch.full((B, k, 2), float("-inf"), device=dev, dtype=dt).scatter_reduce(
            1, idx, torch.where(act, slots_xy, float("-inf")), "amax", include_self=True)
        # a structure with no active slot occupies nothing and is never pushed or pushes
        present = (hi[..., 0] >= lo[..., 0])
        lo = torch.where(present.unsqueeze(-1), lo, torch.zeros_like(lo))
        hi = torch.where(present.unsqueeze(-1), hi, torch.zeros_like(hi))
        half = 0.5 * (hi - lo) + 0.5 * self.pitch + 0.02
        ctr = 0.5 * (hi + lo)
        shift = torch.zeros(B, k, 2, device=dev, dtype=dt)
        room = (self.placeable_xy - (half - 0.5 * self.pitch - 0.02)).clamp_min(0.0)
        for _ in range(16):
            shift = (ctr + shift).clamp(-room, room) - ctr
            for j in range(1, k):
                for i in range(j):
                    d = (ctr[:, j] + shift[:, j]) - (ctr[:, i] + shift[:, i])
                    pen = (half[:, j] + half[:, i]) - d.abs()
                    ov = (pen > 0).all(dim=-1) & present[:, i] & present[:, j]
                    ax = (pen[:, 1] < pen[:, 0]).long()
                    dax = d.gather(1, ax[:, None])[:, 0]
                    sgn = torch.where(dax >= 0, 1.0, -1.0).to(dt)
                    amt = (0.5 * pen.gather(1, ax[:, None])[:, 0] + 1e-3) * sgn * ov.to(dt)
                    shift[:, j] = shift[:, j].scatter_add(1, ax[:, None], amt[:, None])
                    shift[:, i] = shift[:, i].scatter_add(1, ax[:, None], -amt[:, None])
        shift = (ctr + shift).clamp(-room, room) - ctr
        return shift[:, sid]

    def place_goals(self, rest_obj: torch.Tensor, arange_n: torch.Tensor,
                    active_mask: torch.Tensor) -> torch.Tensor:
        n = rest_obj.shape[1]
        dev, dt = rest_obj.device, rest_obj.dtype
        off, _, _ = self.scene_layout(active_mask, dt)
        key = (n, str(dev))
        if key not in self._anchor_cache:
            self._anchor_cache[key] = torch.as_tensor(self.anchor_of(n), device=dev,
                                                      dtype=torch.long)
        anchor = rest_obj[:, self._anchor_cache[key], :3]
        out = torch.zeros_like(rest_obj)
        out[..., :3] = anchor + off
        out[..., :2] = out[..., :2] + self._separate(out[..., :2], n, active_mask)
        L = self.placeable_xy
        slots = out[..., :2]
        act = active_mask.unsqueeze(-1)
        lo = torch.where(act, slots, float("inf")).amin(dim=1)
        hi = torch.where(act, slots, float("-inf")).amax(dim=1)
        fit = (-L - lo).clamp_min(0.0) - (hi - L).clamp_min(0.0)
        shift = torch.where(((hi - lo) > 2.0 * L), -0.5 * (lo + hi), fit)
        out[..., :2] = slots + shift[:, None, :]
        out[..., 2] = self.rest_z + off[..., 2]
        out[..., 3] = 1.0
        return out

    def fits(self, n: int) -> bool:
        """Every structure's slot centres span at most the placeable square. A structure wider
        than that is centred by `place_goals` and its end slots can never be reached, so a count
        where this fails measures the arena, not a policy."""
        off = np.asarray(self.slot_offsets(n))[:, :2]
        anchors = self.anchor_of(n)
        return all(np.ptp(off[anchors == a], axis=0).max() <= 2.0 * self.placeable_xy
                   for a in np.unique(anchors))


class WallGoal(LatticeGoal):
    """
    Build a **wall**: a single-thickness slab ``wall_width`` cubes wide, courses stacked in ẑ,
    neighbours flush along the wall axis (x̂). Object ``k`` takes row ``k // W``, column
    ``k % W``, filled bottom-up so a course is complete before the one above it starts.

    Support DAG: ``support(k) = k - W`` (the cube directly below), none for the bottom course
    — a stacked bond, not a running bond. The stacked bond is the honest default here: a
    running bond's half-pitch offset gives every brick TWO supports, which the lattice machinery
    supports (see :class:`PyramidGoal`), but it also makes the bottom course a different width
    from the ones above, which turns "how many objects" into a ragged function of N.

    Count scaling, by which of the two dimensions is fixed:

    * ``wall_width > 0, wall_height = 0`` — ONE wall grows UPWARD as N rises. Height is the
      hard axis, so this is the aggressive setting.
    * ``wall_width > 0, wall_height > 0`` — a wall is capped at ``W * wall_height`` and N
      scales by adding independent WALLS, mirroring :class:`TowersGoal`'s piles.
    * ``wall_width = 0, wall_height > 0`` — ONE wall of ``wall_height`` courses grows LONGER,
      ``W = ceil(N / wall_height)`` wide: a stacked line.

    Why a wall is harder than the equivalent towers. Identical cube count, identical heights —
    the only difference is that the piles now TOUCH. So it isolates exactly one variable: the
    placement clearance. Every approach that was free in a tower now risks the neighbour, and
    the jaw itself needs somewhere to be.
    """

    def __init__(self, wall_width: int, wall_height: int, **kw):
        super().__init__(**kw)
        assert wall_width >= 1 or wall_height >= 1, (
            f"a wall needs a fixed width or a fixed height, got width={wall_width} "
            f"height={wall_height}")
        self.wall_width = wall_width
        self.wall_height = wall_height

    def structure_size(self) -> int:
        return self.wall_width * self.wall_height

    def width_of(self, n: int) -> int:
        """Cubes per course in an n-object episode."""
        return self.wall_width or -(-n // self.wall_height)

    def _rc(self, n):
        """(row, col, width) of every index WITHIN its own wall."""
        s = self.structure_size()
        w = self.width_of(n)
        j = np.arange(n) % s if s > 0 else np.arange(n)
        return j // w, j % w, w

    def slot_offsets(self, n: int) -> np.ndarray:
        r, c, w = self._rc(n)
        off = np.zeros((n, 3))
        off[:, 0] = (c - 0.5 * (w - 1)) * self.pitch
        off[:, 2] = r * self.stack_dz
        return off

    def level_of(self, n: int) -> np.ndarray:
        return self._rc(n)[0]

    def supports_of(self, n: int) -> np.ndarray:
        r, _, w = self._rc(n)
        sup = (np.arange(n) - w).reshape(n, 1)
        sup[r == 0] = -1
        return sup


class PyramidGoal(LatticeGoal):
    """
    Build a **pyramid**: a triangular stack in a vertical plane — a bottom course of
    ``pyramid_base`` cubes, then ``base-1``, and so on to a single cube at the apex. Each
    upper cube nestles in the groove between the two below it, so its slot is offset half a
    pitch from theirs.

    This is the goal that exercises the **two-support DAG**: a cube counts only once BOTH
    bricks under it are placed, and its commanded slot is retargeted to the centroid of where
    those two actually came to rest. A single-support rule would command the apex off-centre
    the moment either support landed a centimetre wide.

    Layout. A pyramid holds ``T = base*(base+1)/2`` cubes; object ``k`` takes position
    ``k mod T`` within pyramid ``k // T``, filled course by course from the bottom. N need not
    fill the triangle — a partial top course is a perfectly well-posed structure — and N
    beyond ``T`` starts an independent second pyramid, so count scales by adding STRUCTURES.

    ``pyramid_base = 1`` degenerates to a single cube on the floor; ``2`` to two cubes plus
    one nestled on top. Both are legitimate small-N members of the same family.

    ``pyramid_base = 0`` is the OTHER count axis: ONE pyramid whose base is the smallest
    ``W`` with ``W(W+1)/2 >= N``, so the count grows a single structure taller and wider
    instead of adding structures. 30 cubes is a base of 8 with five full courses.
    """

    def __init__(self, pyramid_base: int, **kw):
        super().__init__(**kw)
        assert pyramid_base >= 0, f"pyramid_base must be >= 0, got {pyramid_base}"
        self.pyramid_base = pyramid_base

    def _base(self, n: int) -> int:
        if self.pyramid_base > 0:
            return self.pyramid_base
        return int(np.ceil((np.sqrt(8.0 * n + 1.0) - 1.0) / 2.0))

    def structure_size(self) -> int:
        W = self.pyramid_base
        return W * (W + 1) // 2

    def _ri(self, n):
        """(course, position-in-course) of every index WITHIN its own pyramid."""
        W = self._base(n)
        T = W * (W + 1) // 2
        j = np.arange(n) % T
        starts = np.cumsum([0] + [W - r for r in range(W)])[:W]
        r = np.searchsorted(starts, j, side="right") - 1
        return r, j - starts[r]

    def slot_offsets(self, n: int) -> np.ndarray:
        W = self._base(n)
        r, i = self._ri(n)
        off = np.zeros((n, 3))
        off[:, 0] = (i - 0.5 * (W - r - 1)) * self.pitch
        off[:, 2] = r * self.stack_dz
        return off

    def level_of(self, n: int) -> np.ndarray:
        return self._ri(n)[0]

    def supports_of(self, n: int) -> np.ndarray:
        W = self._base(n)
        T = W * (W + 1) // 2
        r, i = self._ri(n)
        starts = np.cumsum([0] + [W - q for q in range(W)])[:W]
        blk = (np.arange(n) // T) * T
        sup = np.full((n, 2), -1, dtype=int)
        dep = r > 0
        lo = starts[np.clip(r - 1, 0, None)]
        sup[dep, 0] = blk[dep] + lo[dep] + i[dep]
        sup[dep, 1] = blk[dep] + lo[dep] + i[dep] + 1
        return sup


class ShapeGoal(LatticeGoal):
    """
    Build a **flat arrangement**: every target sits on the FLOOR, and the targets touch each
    other side to side. Nothing is stacked — but the policy may still grasp and lift, because
    nothing forbids it; lifting is simply not *required*.

    The difficulty here is lateral clearance: the hazard is the neighbour you set down next
    to, so the approach and the placement order both matter.

    With the parallel-jaw gantry the closed jaw plates occupy roughly ``[h, h + jaw_thickness]``
    either side of the carried cube's centre along the jaw axis, and a flush neighbour has its
    face at exactly ``h``. A cube therefore cannot be set down flush against an existing
    neighbour along the jaw axis, and an interior slot with occupied neighbours on both
    jaw-axis sides is unreachable. A filled square is completable only in orders that leave
    each new cube a free side.

    Patterns
    --------
    ``line``   : N in a row along x̂, centred on the anchor.
    ``rect``   : ``shape_cols`` columns × ``ceil(N/cols)`` rows in the floor plane, centred.
    ``square`` : ``rect`` with ``cols = ceil(sqrt(N))`` — as square as N allows.

    Every level is 0 and the support DAG is empty, so there is no gating: objects are
    independent and each simply has to be in its own slot, at rest, on the floor. What couples
    them is physics, not the objective.
    """

    def __init__(self, shape_pattern: str, shape_cols: int, **kw):
        super().__init__(**kw)
        known = ("line", "rect", "square")
        assert shape_pattern in known, \
            f"unknown shape_pattern {shape_pattern!r}; supported: {known}"
        self.shape_pattern = shape_pattern
        self.shape_cols = shape_cols

    def structure_size(self) -> int:
        return 0

    def _cols(self, n: int) -> int:
        if self.shape_pattern == "line":
            return n
        if self.shape_pattern == "square":
            return int(np.ceil(np.sqrt(n)))
        return self.shape_cols

    def slot_offsets(self, n: int) -> np.ndarray:
        off = np.zeros((n, 3))
        c = self._cols(n)
        col, row = np.arange(n) % c, np.arange(n) // c
        off[:, 0] = (col - 0.5 * (c - 1)) * self.pitch
        off[:, 1] = (row - 0.5 * (row.max(initial=0))) * self.pitch
        return off

    def level_of(self, n: int) -> np.ndarray:
        return np.zeros(n, dtype=int)

    def supports_of(self, n: int) -> np.ndarray:
        return np.full((n, 0), -1, dtype=int)


def make_task_goal(goal_cfg: GoalConfig, object: ObjectGeometry,
                   placeable_xy: float) -> StructureGoal:
    """Build a :class:`StructureGoal` from a task's :class:`~wave_rl.config.GoalConfig` and the
    scene's own object geometry — the single extension point: register a new objective by
    adding a branch keyed on ``goal_cfg.type``.

    ``eps_pos`` arrives already resolved (the caller fills an unset value from the frozen
    skill's own tolerance). Every LENGTH the objective needs beyond the tolerances comes from
    ``object``, so a scene that rescales its objects rescales the whole structure with it and
    no task block restates a half-extent or a rest height. ``placeable_xy`` bounds where a
    lattice slot may sit (see :class:`LatticeGoal`).

    Supported types
    ---------------
    ``towers`` : ``ceil(N/tower_height)`` INDEPENDENT piles — piles stand metres apart, so
        lateral placement is relaxed to "anywhere over the support". ``tower_height = 0``
        makes it one pile of N.

    ``wall`` / ``pyramid`` / ``shape`` : CONTACT structures — neighbouring targets touch, so
        the lateral tolerance tightens to a strict fraction of the lattice pitch. ``wall`` is a
        ``wall_width``-wide slab of stacked courses; ``pyramid`` a triangular stack whose every
        brick has TWO supports; ``shape`` a flat floor arrangement (line / rect / square) with
        no stacking at all and the whole difficulty in lateral clearance.
        ``pyramid_base = 0`` grows one pyramid with N instead of adding structures.
    """
    common = dict(object=object, eps_pos=goal_cfg.eps_pos, eps_vel=goal_cfg.eps_vel,
                  stack_dz=goal_cfg.stack_dz, cfg=goal_cfg)
    gtype = goal_cfg.type
    if gtype == "towers":
        return TowersGoal(**common, tower_height=goal_cfg.tower_height)
    print(f"-- GOAL REACHABILITY BOUND: lattice slots are kept inside |xy| <= "
          f"{placeable_xy:.3f} m, the largest |xy| this scene's config pool ever rests an "
          f"object at. A structure that would overhang is translated RIGIDLY back inside "
          f"(pitch/contact/support DAG untouched); one too wide to fit at all is centred.")
    lattice = dict(**common, xy_frac=goal_cfg.lattice_xy_frac, placeable_xy=placeable_xy)
    if gtype == "wall":
        print(f"== LATTICE GOAL: wall  width={goal_cfg.wall_width or 'grows with N'} "
              f"height={goal_cfg.wall_height or 'grows with N'}")
        return WallGoal(**lattice, wall_width=goal_cfg.wall_width,
                        wall_height=goal_cfg.wall_height)
    if gtype == "pyramid":
        print(f"== LATTICE GOAL: pyramid  base={goal_cfg.pyramid_base or 'grows with N'}")
        return PyramidGoal(**lattice, pyramid_base=goal_cfg.pyramid_base)
    if gtype == "shape":
        print(f"== LATTICE GOAL: shape  pattern={goal_cfg.shape_pattern} "
              f"cols={goal_cfg.shape_cols}   (line = N in a row | rect = cols x ceil(N/cols) "
              f"| square = cols from sqrt(N))")
        return ShapeGoal(**lattice, shape_pattern=goal_cfg.shape_pattern,
                         shape_cols=goal_cfg.shape_cols)
    raise ValueError(
        f"unknown goal type {gtype!r}; supported: 'towers', 'wall', 'pyramid', 'shape'. Add "
        f"a new objective by subclassing StructureGoal and registering it here.")
