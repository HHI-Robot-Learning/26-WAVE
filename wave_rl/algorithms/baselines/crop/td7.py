from dataclasses import dataclass

import torch

from .base import CropBaseline, CropBaselineConfig


@dataclass
class TD7CropConfig(CropBaselineConfig):
    #: decisions of uniform random action before the first gradient step; not the frozen
    #: skill's action, since seeding from it would defeat a from-scratch arm
    warmup_decisions: int = 2_000
    updates_per_decision: int = 32


class TD7Crop(CropBaseline):
    """A TD7 actor from scratch on the crop, with no frozen skill: the bar WAVE's action law
    has to clear. The actor reads the whole scene at every count through the scripted pair
    floor and learns nothing about how to look at it.

    The action is re-decided once per decision, the training loop's own cadence."""

    def __init__(self, skill, env, cfg: TD7CropConfig) -> None:
        super().__init__(skill, env, cfg)
        self._actor = None
        self._action = None
        self._n_decisions = 0
        self._warming = True

    def _decision_dim(self) -> int:
        """What the TD7 actor outputs: here the action itself."""
        return self.action_dim

    def _build(self) -> None:
        super()._build()
        if self._actor is None:
            self._actor = self._td7(self._decision_dim())

    def _terminal_value(self, obs_true, state, warp) -> torch.Tensor:
        return self._actor.evaluate_critic(self._flat_crop(obs_true, state))

    def _guided_action(self, obs, warp, identity: bool = False) -> torch.Tensor:
        """The runtime's tick, then the actor's action. With nothing to work on there is no
        falling back to the frozen skill: this arm has none."""
        self._tick(obs)
        return self._action

    def _decide(self, crop: torch.Tensor, explore: bool) -> torch.Tensor:
        """The actor's decision at `crop`; uniform random while a training run warms up."""
        if explore and self._warming:
            return torch.rand(crop.shape[0], self._decision_dim(), device=self.device) * 2.0 - 1.0
        return self._actor.select_action(crop, use_exploration=explore)

    def _collect(self, obs):
        crop = self._flat_crop(obs)
        self._action = self._decide(crop, explore=True)
        obs, reward, done = self._step_window(obs, None, identity=True)
        self._actor.replay_buffer.add(crop, self._action, self._flat_crop(obs), reward,
                                      self.reward.discount * (~done).float())
        self.logger.add("train/mean_decision_reward", reward.mean())
        self._n_decisions += 1
        return obs

    def _run_iteration(self) -> None:
        self._warming = self._n_decisions < self.cfg.warmup_decisions
        super()._run_iteration()

    def _update(self, decisions: int) -> None:
        if self._n_decisions >= self.cfg.warmup_decisions:
            for _ in range(decisions * self.cfg.updates_per_decision):
                self._actor.train(self.logger)
        self.logger.add("train/buffer_size", self._actor.replay_buffer.size)

    def act_batched(self, obs, dones=None):
        if self._state is None:
            self._first_tick(obs)
        if self._greedy_tick % self.cfg.cadence_ticks == 0:
            self._action = self._decide(self._flat_crop(obs), explore=False)
        self._greedy_tick += 1
        self._pending_done = dones
        return self._guided_action(obs, None, identity=True)

    def _checkpoint(self) -> dict:
        return {**super()._checkpoint(), "td7": self._actor.state_dict()}

    def _apply_checkpoint(self, checkpoint: dict) -> None:
        super()._apply_checkpoint(checkpoint)
        self._actor.load_state_dict(checkpoint["td7"])
