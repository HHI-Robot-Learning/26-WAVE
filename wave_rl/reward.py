import math

import torch


class TaskMeasure:
    """sat and Phi, recomputed EVERY tick from the observation. Never latched — a knocked
    object un-satisfies within a tick.

        sat_i      the object is satisfied by the TASK's own predicate
        Phi_sparse = sum_i sat_i                              (active objects only)
        Phi_dense  = Phi_sparse + c_d * sum_i closeness(|x_i - g_i|)

    `c_d < 1` keeps one object completion (+1) dominant over any distance rearrangement.
    Disturbance is signed by the task itself: knocking a placed object drops Phi_sparse by
    1 within a tick, so there is no separate do-not-disturb term anywhere in the system.

    `sat` delegates to the task goal rather than re-implementing it. A re-implementation would
    be position-only, so on a rest-at-goal task it would score an object rolling through its
    goal as satisfied, and on a structure task it would drift from the predicate the episode
    is scored on.
    """

    def __init__(self, cfg, goal, h: float) -> None:
        self.cfg = cfg
        self.goal = goal
        self.h = float(h)

    def closeness(self, d: torch.Tensor) -> torch.Tensor:
        """distance -> potential, THE single definition. Every dense Phi term routes through here so
        they cannot drift apart.

            log(D/(d+d0)) / log(D/d0),  clamped to [0, 1],  d0 = the task's own eps_pos

        dPhi/dd ~ 1/d: every OCTAVE of approach pays the same, 4 m -> 2 m or 14 cm -> 7 cm. It is
        the only form here invariant to arena scale, and that matters at BOTH ends -- a form that
        saturates leaves distant pairs with no gradient at all, while near the goal the final 7 cm
        of a placement has to stay above the one-tick measure noise floor, which is where the
        placement must be precise.
        """
        d0 = self.goal.eps_pos
        D = self.cfg.phi_closeness_scale_h * self.h
        return (torch.log(D / (d + d0)) / math.log(D / d0)).clamp(0.0, 1.0)

    def supported(self, obs, rest_z: torch.Tensor) -> torch.Tensor:
        """(B, N) bool -- the object currently RESTS on something. The task goal's own universal
        statics predicate, against the reset-state minimum object height FROZEN per episode."""
        return self.goal.supported(obs.obj_feat.float(), obs.active_mask > 0.5, rest_z)

    def sat(self, obs) -> torch.Tensor:
        """(B, N) bool — the task's own predicate, re-evaluated every tick, never latched.

        It is NOT the env's `satisfied_mask`, which is one tick stale and zero at reset — a
        PRE-BUILT structure would read unsatisfied there and get targeted and toppled.
        """
        speeds = torch.linalg.norm(obs.obj_qvel[..., :3].float(), dim=-1)
        return self.goal.satisfied(obs.obj_feat.float(), obs.goal_feat.float(), speeds,
                                   obs.active_mask > 0.5).bool()

    def phi(self, obs, sat: torch.Tensor):
        """(Phi_sparse, Phi_dense) — what the REWARD reads."""
        active = obs.active_mask > 0.5
        phi_sparse = (sat & active).float().sum(dim=1)
        d = torch.linalg.norm(obs.obj_feat.float() - obs.goal_feat.float(), dim=-1)
        phi_dense = phi_sparse + self.cfg.phi_closeness_weight * (
            self.closeness(d) * active.float()).sum(dim=1)
        return phi_sparse, phi_dense


class SceneReward:
    """THE reward every learner trains on, one env tick at a time.

    Per env tick, with gamma the scene's discount per DECISION and gamma_tick its per-tick root:

        d_phi  = gamma_tick * Phi(s') - Phi(s)                     [PBRS]
                 - gamma_tick * Phi(s') on ends that do NOT bootstrap  [terminal reclaim]
        r     += d_phi
        r     += w * (1-gamma)/k * (#satisfied / #active)          [occupancy income]
        r     += w * gamma            on termination               [success-absorbing]
        r     += w * b / #active      per object, FIRST time only  [reaching term]

    Phi is the closeness part of `TaskMeasure.phi` alone: the #satisfied level is paid
    directly as the objective, so keeping it inside Phi too would double-count it as shaping.
    Each tick's Phi is measured against THAT tick's goals, so a goal that moves with its
    realized support is charged like any other change of state and the shaping telescopes:
    no closed loop of motion nets reward. Occupancy income prices holding a placement
    mid-episode, which a plain dPhi return leaves nearly unpriced. The reaching term is paid
    once per object per episode, so place-knock-place is unfarmable.

    A learner that bootstraps a time limit from its own critic adds that value itself; it is
    the one term no reward can compute without the learner.

    Usage: `reset(obs)` once at the start of a rollout, then `step(...)` after every env
    tick and `refresh(obs)` before the next decision. `discount` stays the discount of the
    episode the last `step` belonged to until `refresh` derives the next episode's.
    """

    def __init__(self, measure: TaskMeasure, cfg, discount_of, decision_ticks: int) -> None:
        self.measure = measure
        self.cfg = cfg
        self.discount_of = discount_of
        self.decision_ticks = decision_ticks
        self.discount = None
        self.terms = {}

    def reset(self, obs) -> None:
        """Fresh per-scene state for `obs`'s batch: no latch, nothing ever satisfied, the
        discount of each scene's episode and Phi measured at `obs`."""
        active = obs.active_mask
        self._hyst_on = torch.zeros_like(active, dtype=torch.long)
        self._hyst_off = torch.zeros_like(active, dtype=torch.long)
        self._hyst_latch = torch.zeros_like(active, dtype=torch.bool)
        self._ever_satisfied = torch.zeros_like(active, dtype=torch.bool)
        self.refresh(obs)
        sat = self.measure.sat(obs)
        self._phi_prev = self._shaping_potential(obs, self._latched_sat(sat, advance=False))

    def refresh(self, obs) -> None:
        """Each scene's discount from its own object count, from the fixed horizon
        `h = (1-gamma)*D`. Income and bonus MUST scale together, or "finishing strictly
        dominates hovering one short" breaks."""
        k = self.decision_ticks
        self.discount = self.discount_of(obs.active_mask)
        self.tick_discount = self.discount ** (1.0 / k)
        scale = self.cfg.occupancy_weight_scale
        self.occupancy_weight = scale * (1.0 - self.discount) / k
        self.success_bonus = scale * self.discount

    def step(self, obs_true, obs_next, terminated: torch.Tensor, truncated: torch.Tensor,
             info: dict) -> torch.Tensor:
        """(B,) the reward of one env tick. `obs_true` is the tick's own end state (the
        terminal state of a finished episode), `obs_next` what the learner acts from next
        (already reset where an episode ended). The per-term split lands in `terms`."""
        c = self.cfg
        sat = self.measure.sat(obs_true)
        latched = self._latched_sat(sat)
        phi = self._shaping_potential(obs_true, latched)

        live = (~info["sim_failure"]).float()
        reclaim = (terminated | info["unachievable"]).float()
        d_phi = self.tick_discount * phi - self._phi_prev
        d_phi = d_phi - self.tick_discount * phi * reclaim
        shaping = d_phi * live

        n_active = obs_true.active_mask.float().sum(dim=1).clamp_min(1.0)
        income = self.occupancy_weight * (sat.float().sum(dim=1) / n_active) * live
        success = torch.where(terminated, self.success_bonus, torch.zeros_like(income)) * live

        # Gated on the debounced latch, not raw sat: sat is position plus a purely geometric
        # support test (the floor check is a height check), so a carried object swept through
        # a ground slot at goal height reads satisfied for one tick and would collect the
        # whole bonus without ever being released.
        fresh = latched & ~self._ever_satisfied & obs_true.active_mask.bool()
        placement = (c.occupancy_weight_scale * c.first_completion_bonus
                     * fresh.float().sum(dim=1) / n_active * live)
        self._ever_satisfied |= latched

        finished = terminated | truncated
        self._ever_satisfied.masked_fill_(finished.unsqueeze(1), False)
        sat_next = self.measure.sat(obs_next)
        self._phi_prev = self._shaping_potential(
            obs_next, self._latched_sat(sat_next, reset=finished, advance=False))
        self.terms = {"shaping": shaping, "occupancy": income, "success": success,
                      "placement": placement}
        return shaping + (income + success) + placement

    def _latched_sat(self, raw: torch.Tensor, reset=None, advance: bool = True):
        """Debounced `sat`. Mutates the latch IN PLACE — never rebind these tensors.
        `advance=False` reads without stepping the counters, because Phi is evaluated twice
        per env tick."""
        k = self.cfg.sat_hysteresis_ticks
        if reset is not None:
            self._hyst_on.masked_fill_(reset.unsqueeze(1), 0)
            self._hyst_off.masked_fill_(reset.unsqueeze(1), 0)
            self._hyst_latch.copy_(torch.where(reset.unsqueeze(1), raw, self._hyst_latch))
        if advance:
            self._hyst_on.copy_(torch.where(raw, self._hyst_on + 1,
                                            torch.zeros_like(self._hyst_on)))
            self._hyst_off.copy_(torch.where(raw, torch.zeros_like(self._hyst_off),
                                             self._hyst_off + 1))
            self._hyst_latch.copy_(torch.where(
                self._hyst_on >= k, torch.ones_like(self._hyst_latch),
                torch.where(self._hyst_off >= k, torch.zeros_like(self._hyst_latch),
                            self._hyst_latch)))
        return self._hyst_latch.clone()

    def _shaping_potential(self, obs, sat: torch.Tensor) -> torch.Tensor:
        sparse, dense = self.measure.phi(obs, sat)
        return dense - sparse
