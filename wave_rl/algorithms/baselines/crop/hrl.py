from dataclasses import dataclass

import torch

from .td7 import TD7Crop, TD7CropConfig

#: the manager's decision: a bounded xyz offset from the worked object's true goal
SUBGOAL_DIM = 3


@dataclass
class HRLConfig(TD7CropConfig):
    #: R of the subgoal offset ball around the object's true goal, in units of h: a local
    #: correction, not a place to send the object somewhere unrelated to its slot
    subgoal_radius_h: float = 0.75
    exploration_noise: float = 0.03
    updates_per_decision: int = 1


class HRL(TD7Crop):
    """A TD7 manager over the crop sets subgoals for the frozen skill, the low level of the
    hierarchy and the only thing that touches the actuators.

    The manager decides a 3-D offset; the skill chases the worked object's true goal shifted
    by it, re-acting every tick from its own observation. Offset 0 is the frozen-skill
    baseline, so whatever the manager buys over it has to be learned.

    The option is adaptive: an object's offset is drawn when the runtime starts working it
    (or its episode resets) and held for as long as the runtime keeps working it, since a
    skill trained on one fixed goal per episode cannot be expected to chase a jittering one."""

    def __init__(self, skill, env, cfg: HRLConfig) -> None:
        super().__init__(skill, env, cfg)
        self._offset = None
        self._committed = None
        self._force = None

    def _decision_dim(self) -> int:
        return SUBGOAL_DIM

    def _decide(self, crop: torch.Tensor, explore: bool) -> torch.Tensor:
        """The held offset, redrawn where the worked object changed or an episode reset."""
        candidate = super()._decide(crop, explore)
        atom = self._state.atom
        if self._offset is None:
            self._offset = torch.zeros_like(candidate)
            self._committed = torch.full_like(atom, -1)
            self._force = torch.zeros_like(atom, dtype=torch.bool)
        changed = (atom != self._committed) | self._force
        self._offset = torch.where(changed.unsqueeze(1), candidate, self._offset)
        self._committed = torch.where(changed, atom, self._committed)
        self._force = torch.zeros_like(self._force)
        return self._offset

    def _guided_action(self, obs, warp, identity: bool = False) -> torch.Tensor:
        """The runtime's tick, then the frozen skill toward the subgoal of the object that
        tick committed to; the runtime's own action where there is nothing to work on."""
        naive = self._tick(obs)
        state = self._state
        rows = torch.arange(obs.active_mask.shape[0], device=self.device)
        subgoal = (obs.goal_feat[rows, state.atom].float()
                   + self.cfg.subgoal_radius_h * self.h * self._action)
        action = self.skill.act(self.skill.present(obs, state.atom, goal_override=subgoal))
        return torch.where(state.has_atom.unsqueeze(1), action, naive)

    def _step_window(self, obs, warp, identity: bool):
        obs, reward, done = super()._step_window(obs, warp, identity)
        self._force = done
        return obs, reward, done

    def act_batched(self, obs, dones=None):
        if self._force is not None:
            self._force |= dones
        return super().act_batched(obs, dones)

    def reset(self) -> None:
        super().reset()
        self._offset = None
        self._committed = None
        self._force = None
