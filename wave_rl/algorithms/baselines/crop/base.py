from dataclasses import dataclass

import torch

from wave_rl.td7.agent import TD7Agent, TD7Config

from wave_rl.algorithms.crop import CROP_CORE, PAIR_DIM, pair_floor
from wave_rl.algorithms.wave.algorithm import WaveAlgorithm
from wave_rl.algorithms.wave.config import WaveConfig

#: what a flat-state actor reads: the fixed core about the worked object and the robot, plus
#: the scripted floor of the pair rows over every other object
STATE_DIM = CROP_CORE + PAIR_DIM


@dataclass
class CropBaselineConfig(WaveConfig):
    """A learner on WAVE's crop, decision clock and scene reward. Nearest-first picks the
    object, so the action law is the only thing the arm learns."""

    #: the crop must not carry the frozen skill's output
    crop_tool_action: bool = False
    selector_mode: str = "analytic"
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    exploration_noise: float = 0.1
    updates_per_decision: int = 1


class CropBaseline(WaveAlgorithm):
    """WAVE's runtime, reward and episode clock under another action law.

    Everything that is not the action law is inherited, which is what makes the arms
    comparable to WAVE. The parent also builds a warp head that is never sampled here."""

    def _tick(self, obs) -> torch.Tensor:
        """The runtime's tick (selection, episode refresh): the frozen skill's action on the
        object it commits to."""
        dones = self._pending_done
        self._pending_done = None
        return super(WaveAlgorithm, self).act_batched(obs, dones=dones)

    def _first_tick(self, obs) -> None:
        """A deployment's first tick, so the runtime commits to an object before the first
        crop is read."""
        super(WaveAlgorithm, self).act_batched(obs)

    def _flat_crop(self, obs, state=None) -> torch.Tensor:
        """(B, STATE_DIM) the crop as one vector: the core, then the scripted pair floor."""
        crop = self.build_crop(obs, self._state if state is None else state)
        return torch.cat([crop.core, pair_floor(crop.pairs, crop.mask)], dim=1)

    def _td7(self, action_dim: int) -> TD7Agent:
        """A TD7 agent over the flat crop, with this arm's learning settings."""
        c = self.cfg
        hp = TD7Config(learning_rate=c.learning_rate, batch_size=c.batch_size,
                       buffer_size=c.buffer_size, exploration_noise=c.exploration_noise,
                       value_clip=[-10.0, 30.0])
        return TD7Agent(state_dim=STATE_DIM, action_dim=action_dim, hp=hp)

    def _collect(self, obs):
        """One decision of the training rollout -> the next observation."""
        raise NotImplementedError

    def _update(self, decisions: int) -> None:
        """The gradient steps after `decisions` collected decisions."""
        raise NotImplementedError

    def _run_iteration(self) -> None:
        c = self.cfg
        obs = self._obs
        self._collecting = True
        with torch.no_grad():
            for _ in range(c.decisions_per_iteration):
                obs = self._collect(obs)
        self._collecting = False
        self._obs = obs
        self._update(c.decisions_per_iteration)
        self._iteration += 1
        self.total_steps += c.decisions_per_iteration * c.cadence_ticks * self.B

    def load(self, path: str) -> None:
        """An eval-only replay builds the networks before the weights land in them."""
        self._build()
        super().load(path)
