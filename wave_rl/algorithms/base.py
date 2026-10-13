from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import torch

from wave_rl.reward import SceneReward, TaskMeasure


@dataclass
class AlgorithmConfig:
    """Base config for an :class:`Algorithm` — the typed, default-bearing schema its
    hyperparameters live in. A param-less algorithm uses this empty base as-is; one with
    knobs subclasses it with fields + defaults.

    The harness loads a task's ``algorithm_cfg`` onto this schema with OmegaConf, so a task file
    only ever names overrides and unknown keys / wrong types fail at startup. Restrict
    fields to yaml-native types (no ``Callable``); express choices as strings.

    Every algorithm trains on the one :class:`~wave_rl.reward.SceneReward`, so its knobs live
    here, on the base, and a learner never restates them.
    """

    #: w; scales both the per-tick occupancy income and the success bonus together (their
    #: ratio is fixed by the episode clock, not this knob).
    occupancy_weight_scale: float = 4.0
    #: b, the reaching bonus; paid w*b/n_active on the first tick each object is satisfied,
    #: once per episode (prevents farming placement/knockdown cycles). Interacts strongly
    #: with the discount horizon.
    first_completion_bonus: float = 0.65
    #: c_d, the dense closeness weight; keep < 1 so atom completion (+1) dominates distance
    #: rearrangement.
    phi_closeness_weight: float = 0.5
    #: D in log(D/(d+d0))/log(D/d0), in units of h; scale-invariant across arena size.
    phi_closeness_scale_h: float = 62.5
    #: h = (1-gamma)*D, D the episode length in decisions: the discount is derived per
    #: episode from this one horizon.
    discount_horizon: float = 3.0
    #: Debounce (in ticks) on the `sat` signal the reaching bonus reads, so a momentary
    #: false-positive on_floor read cannot trigger it.
    sat_hysteresis_ticks: int = 5


class Algorithm(ABC):
    """A controller the benchmark can train on one environment and roll out on another.

    The contract is deliberately small — the four things the harness needs from any
    controller, whether it learns end to end, distils from a teacher, or wraps a frozen
    single-object skill: :meth:`learn` against :attr:`env` (a no-op for a zero-shot
    baseline, so trained and zero-shot controllers flow through one harness),
    :meth:`act_batched` for the rollout, and :meth:`save` / :meth:`load`.

    An algorithm carries no registry metadata: its CLI name and its config class live in
    :data:`~wave_rl.algorithms.ALGORITHMS`, not on the class, so a concrete algorithm
    is just its behaviour.

    Every algorithm owns the benchmark's one reward, :attr:`reward`, and the discount it is
    defined under, :meth:`discount`. A learner trains on exactly that reward with exactly that
    discount; a zero-shot controller never touches it, so it is never built.
    """

    def __init__(self, env, cfg: AlgorithmConfig) -> None:
        self.env = env
        self.cfg = cfg
        self.logger = None
        #: Total env steps trained so far — the re-entrant :meth:`learn` advances it; the
        #: harness reads it to detect a non-learning algorithm (it stays 0).
        self.total_steps = 0
        #: The run's whole step budget, set by the harness before the first block. Anything
        #: scheduled over the length of training reads it.
        self.budget = 0

    @cached_property
    def measure(self) -> TaskMeasure:
        """The task's own predicates and closeness, as every reward and runtime reads them."""
        return TaskMeasure(self.cfg, self.env.task_goal, self.env.scene.object.half_extent)

    @cached_property
    def reward(self) -> SceneReward:
        """The scene reward this algorithm trains on, built on first use."""
        return SceneReward(self.measure, self.cfg, self.discount, self.decision_ticks)

    @property
    def decision_ticks(self) -> int:
        """Env ticks per decision of this learner -- the clock its discount is stated on."""
        return 1

    def discount(self, active_mask: torch.Tensor) -> torch.Tensor:
        """(B,) gamma per DECISION for each scene's episode, from the fixed horizon
        `h = (1-gamma)*D`.

        D = n * time_per_object / (dt * decision_ticks) is the episode length in decisions, so
        holding h fixed means the end of the episode is worth exp(-h) of its face value no
        matter how many objects are in the scene or how often the learner decides. A fixed
        scalar gamma does not: the terminal bonus decays exponentially with episode length,
        so at high object counts the success-absorbing bonus is effectively gone.
        """
        env = self.env
        n = active_mask.float().sum(dim=1).clamp_min(1.0)
        seconds = (env.time_per_object * n if env.time_per_object is not None
                   else torch.full_like(n, env.episode_time_limit))
        decisions = seconds / (env.tau_step * self.decision_ticks)
        return (1.0 - self.cfg.discount_horizon / decisions).clamp(0.5, 0.99999)

    @abstractmethod
    def learn(self, n_steps: int) -> None:
        """Train against :attr:`env` for ``n_steps`` more environment steps.

        Re-entrant: the harness calls this repeatedly in blocks, so an implementation must
        persist its state (replay buffer, networks, current observation, step counter)
        across calls. A zero-shot baseline implements it as a no-op.
        """

    @abstractmethod
    def act_batched(self, obs, dones: torch.Tensor | None = None) -> torch.Tensor:
        """(B, action_dim) on device for a batched observation.

        ``dones`` marks scenes whose previous step ended an episode, so ``obs`` is already
        their fresh reset observation. An algorithm carrying per-episode state must be given
        it, or that state silently describes the previous episode.
        """

    def reset(self) -> None:
        """Drop per-rollout state. No-op by default."""

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement save()")

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement load()")
