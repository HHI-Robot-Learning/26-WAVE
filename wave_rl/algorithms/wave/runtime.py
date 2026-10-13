import math

import torch

from wave_rl.algorithms.lifter import Lifter
from wave_rl.algorithms.crop import CROP_CORE, N_CLEARANCE, PAIR_DIM, Crop, CropBatch, carried_flag
from wave_rl.env.observation import BatchedObservation


class RuntimeState:
    """Per-rollout batched state. One instance per training loop, one per eval context."""

    def __init__(self, B: int, N: int, device) -> None:
        self.B, self.N = B, N
        z = lambda *s, **k: torch.zeros(*s, device=device, **k)   # noqa: E731
        #: the reset-state minimum object height, FROZEN per episode
        self.rest_z = z(B)
        self.atom = z(B, dtype=torch.long)
        self.has_atom = z(B, dtype=torch.bool)
        #: re-selection cooldown after a completion, so the settle bounce cannot re-elect it
        self.atom_cooldown = z(B, N, dtype=torch.long)
        self.atom_ever_sat = z(B, N, dtype=torch.bool)
        #: last tick's satisfaction, so a knock is `was & ~is` — the disturbance trigger
        self.prev_sat = z(B, N, dtype=torch.bool)
        self.tick_in_episode = z(B, dtype=torch.long)


def selection_pool(pending: torch.Tensor, exclude: torch.Tensor) -> torch.Tensor:
    """The set an object is actually chosen FROM: the pending objects minus the step-aside
    mask, falling back to every pending object where the step-aside would leave nothing.
    A learned selector scores exactly the pool the analytic rule would have scored."""
    alternative = (pending & ~exclude).any(dim=1)
    return torch.where(alternative.unsqueeze(1), pending & ~exclude, pending)


def nearest_first(obs, pool: torch.Tensor) -> torch.Tensor:
    """The analytic rule: the pooled object nearest the end effector in xy. xy only, since
    the end effector's z and an object's z live in different frames on the gantry."""
    ee_xy = obs.ee_position[..., :2].float()
    d = torch.linalg.norm(obs.obj_feat[..., :2].float() - ee_xy[:, None, :], dim=-1)
    return torch.where(pool, d, torch.full_like(d, 1e6)).argmin(dim=1)


class WaveRuntime(Lifter):
    """The non-learned per-tick runtime: the measure, the crop, the decision clock, the
    selection latch and the episode bookkeeping.

    The two clocks are not the same:

        warp      re-decided every `cadence_ticks` env ticks, UNCONDITIONALLY
        selection re-decided on `event`, and every event re-decides

        event = completed | lost | cadence

    THERE IS NO EXIT RULE AND NO LATCH. `completed` is the task's own predicate on the
    worked object; `lost` is that same predicate flipping 1 -> 0 on any object, i.e. a
    knock; `cadence` is a fixed period in TASK TIME (`selector_decide_fraction` of
    `time_per_object`). Nothing here references any object except through the task
    predicate, so travelling to a far object is not read as a regression.

    The pool is every active, unsatisfied object, and it contains the incumbent: letting go
    is the selection choosing someone else, and "keep going" is a decision the policy is
    graded on. A false event therefore costs one extra selector pass and nothing else. The
    episode timeout is `n_objects * time_per_object`.

    `WaveAlgorithm` subclasses this and is the only thing that trains.
    """

    def __init__(self, skill, env, cfg) -> None:
        super().__init__(skill, env, cfg)
        self.device = env.device
        self.goal = env.task_goal
        self.h = env.scene.object.half_extent
        self.action_dim = env.action_dim
        self.obs_dim = skill.obs_dim
        self._state = None

        self.crop = Crop(self.h, cfg.crop_radius_h * self.h or skill.disc_radius, skill.layout,
                         cfg.crop_length_scale_h, cfg.crop_length_outer)
        print(self.crop.banner())

        # An object dropped from goal height needs a few free-fall times to settle; the latch
        # must not re-elect it inside that window or the settle bounce reads as a fresh pick.
        # A time in seconds, turned into ticks of THIS plant, with a floor.
        sc = cfg.selector
        tick = env.tau_step
        fall = sc.settle_fall_times * math.sqrt(2.0 * self.goal.eps_pos / 9.81)
        settle = max(int(round(sc.settle_floor_seconds / tick)), int(math.ceil(fall / tick)))
        self._reselect_cooldown = sc.cooldown_settles * settle

        # THE DECISION PERIOD, in env ticks, as a fraction of the task's own per-object time
        # budget: one constant, in a unit every task declares.
        per_object = (env.time_per_object if env.time_per_object is not None
                      else env.episode_time_limit)
        self._decide_ticks = int(round(cfg.selector_decide_fraction * per_object
                                       / env.tau_step))
        print(f"-- DECISION CLOCK  every {self._decide_ticks} env ticks "
              f"({self._decide_ticks * env.tau_step:.2f} s = "
              f"{cfg.selector_decide_fraction:.3g} x {per_object:.1f} s per object), "
              f"plus completion and knock events")

    def present(self, obs, atom, warp=None):
        return self.skill.present(obs, atom, warp)

    def _fresh_state(self, obs: BatchedObservation) -> RuntimeState:
        B, N = obs.active_mask.shape
        state = RuntimeState(B, N, self.device)
        self._reset_scenes(state, obs,
                           torch.ones(B, dtype=torch.bool, device=self.device))
        return state

    def _reset_scenes(self, state: RuntimeState, obs: BatchedObservation,
                      done: torch.Tensor) -> None:
        """Episode-boundary refresh: the frozen rest height, the selection latch and the
        decision clock."""
        active = obs.active_mask > 0.5
        floor = torch.where(active, obs.obj_feat[..., 2].float(),
                            torch.full_like(obs.obj_feat[..., 2].float(), 1e6)).min(dim=1).values
        state.rest_z = torch.where(done, floor, state.rest_z)
        state.has_atom = state.has_atom & ~done
        state.atom_cooldown = torch.where(done.unsqueeze(1),
                                          torch.zeros_like(state.atom_cooldown),
                                          state.atom_cooldown)
        state.tick_in_episode = torch.where(done, torch.zeros_like(state.tick_in_episode),
                                            state.tick_in_episode)
        sat = self.measure.sat(obs)
        state.atom_ever_sat = torch.where(done.unsqueeze(1), sat, state.atom_ever_sat)
        state.prev_sat = torch.where(done.unsqueeze(1), sat, state.prev_sat)

    def tick(self, obs: BatchedObservation, state: RuntimeState) -> torch.Tensor:
        """One control tick. Returns the UNWARPED action; a subclass that owns a warp
        re-runs the skill under it (see `WaveAlgorithm._guided_action`)."""
        rows = torch.arange(obs.active_mask.shape[0], device=self.device)

        sat = self.measure.sat(obs)
        active = obs.active_mask > 0.5
        state.atom_ever_sat = state.atom_ever_sat | (sat & active)
        state.atom_cooldown = (state.atom_cooldown - 1).clamp(min=0)

        # Reads sat DIRECTLY, with no dwell: a dwell delays the exit while the skill keeps
        # commanding the placed object's goal through the settle window and re-grips it.
        # Settle flicker is handled at the SELECTOR, via the re-selection cooldown.
        completed = state.has_atom & sat[rows, state.atom]
        # A KNOCK: anything that was satisfied last tick and is not now. Undebounced on
        # purpose -- a spurious event costs one selector pass, and the incumbent is simply
        # re-elected by its own stickiness.
        lost = (state.prev_sat & ~sat & active).any(dim=1)
        state.prev_sat = sat & active

        event = completed | lost | ((state.tick_in_episode % self._decide_ticks) == 0)
        if bool(event.any()):
            self._decision(obs, state, event, completed, sat, lost)

        action = self.skill.act(self.present(obs, state.atom))
        state.tick_in_episode += 1
        return action

    def _prepare_selection(self, obs, state, sat) -> None:
        """Hook: a learned selector runs ONE pass here per decision tick. Nothing to prepare
        for the analytic rule."""

    def _select(self, obs, state, pool, redo, completed, fresh) -> torch.Tensor:
        """WHICH pooled object to work next. The default is the analytic rule.

        The pool CONTAINS the incumbent, so re-electing it is how a commitment continues and
        there is no separate exit. A subclass may override to score the candidates instead;
        the pool and the `redo` gate are not its business. `fresh` marks the rows whose
        decision begins a new chain (nothing to continue), the only place a training-time
        exploration floor may be charged.
        """
        has = pool.any(dim=1)
        return torch.where(redo & has, nearest_first(obs, pool), state.atom)

    def _decision(self, obs, state, event, completed, sat, lost) -> None:
        """Re-decide over a pool that INCLUDES the incumbent."""
        rows = torch.arange(obs.active_mask.shape[0], device=self.device)
        self._prepare_selection(obs, state, sat)
        cooldown = state.atom_cooldown[rows, state.atom]
        state.atom_cooldown[rows, state.atom] = torch.where(
            completed & event, torch.full_like(cooldown, self._reselect_cooldown), cooldown)
        pending = (obs.active_mask > 0.5) & ~sat
        pool = selection_pool(pending, state.atom_cooldown > 0)
        # A chain STARTS where there is no incumbent left to continue: it completed, it was
        # knocked out from under us, or there never was one. Those are the decisions that
        # carry the exploration floor, so exploration is priced per COMMITMENT rather than
        # per decision.
        fresh = event & (completed | lost | ~state.has_atom)
        state.atom = self._select(obs, state, pool, event, completed, fresh)
        state.has_atom = torch.where(event, pending.any(dim=1), state.has_atom)

    def build_crop(self, obs, state: RuntimeState) -> CropBatch:
        """The crop at the currently committed object: a fixed core about that object and the
        robot, and one geometric row per other object for whoever reads it to reduce."""
        rows = torch.arange(obs.active_mask.shape[0], device=self.device)
        supported_all = self.measure.supported(obs, state.rest_z)
        sat = self.measure.sat(obs)
        carried = carried_flag(obs, state.atom, supported_all, self.h)
        return self.crop.build(
            obs, state.atom, sat, carried,
            supported_all[rows, state.atom], self.tool_action(obs, state.atom))

    @torch.no_grad()
    def tool_action(self, obs, atom: torch.Tensor):
        """pi_0's action at the UNWARPED input, or None when observing the tool is off.

        Deliberately RECOMPUTED rather than reusing the tick's naive action: the crop is
        built at the START of a decision, so a cached value would be one full decision
        stale — and the gantry's ring period is the same order as that lag, which is exactly
        the thing this signal exists to see.
        """
        if not self.cfg.crop_tool_action:
            return None
        return self.skill.act(self.present(obs, atom))

    def act_batched(self, obs: BatchedObservation,
                    dones: torch.Tensor | None = None) -> torch.Tensor:
        """`dones` (B,) marks scenes whose previous step ended an episode, so `obs` is already
        their fresh reset observation and their decision state starts over."""
        if self._state is None:
            self._state = self._fresh_state(obs)
        elif dones is not None:
            self._reset_scenes(self._state, obs, dones)
        return self.tick(obs, self._state)

    def reset(self) -> None:
        self._state = None

    def save(self, path: str) -> None:
        torch.save(self._checkpoint(), path)

    def load(self, path: str) -> None:
        self._apply_checkpoint(torch.load(path, weights_only=False,
                                          map_location=self.device))

    def _checkpoint(self) -> dict:
        return {"dims": {"obs_dim": self.obs_dim, "action_dim": self.action_dim,
                         "crop_core": CROP_CORE, "pair_dim": PAIR_DIM,
                         "n_clearance": N_CLEARANCE},
                "frozen_agent": self.skill.path,
                "disc_radius": self.skill.disc_radius}

    def _apply_checkpoint(self, checkpoint: dict) -> None:
        """The runtime carries no learned state; a subclass that does extends this. The crop
        widths are the one thing it does check: every learned width downstream is derived
        from them, so a head trained on a different crop would load into shapes that happen
        to match and read columns that mean something else."""
        dims = checkpoint["dims"]
        want = {"crop_core": CROP_CORE, "pair_dim": PAIR_DIM, "n_clearance": N_CLEARANCE}
        if {k: dims[k] for k in want} != want:
            raise SystemExit(
                f"this head was trained on a crop of {dims} and this code builds {want}; a "
                f"head is not replayable across a crop layout change.")
