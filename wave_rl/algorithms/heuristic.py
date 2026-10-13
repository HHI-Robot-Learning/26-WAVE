import torch

from wave_rl.algorithms.lifter import Lifter
from wave_rl.algorithms.wave.runtime import nearest_first


def scripted_selection(measure, obs) -> torch.Tensor:
    """The baselines' scripted selection: the active, unsatisfied object nearest the end
    effector. A scene with nothing left unsatisfied keeps its active set, so the rule never
    stalls."""
    active = obs.active_mask > 0.5
    pending = active & ~measure.sat(obs)
    return nearest_first(obs, torch.where(pending.any(dim=1, keepdim=True), pending, active))


class SkillAlone(Lifter):
    """The frozen skill with no training, driven by the scripted selection."""

    def act_batched(self, obs, dones: torch.Tensor | None = None) -> torch.Tensor:
        return self.skill.act(self.skill.present(obs, scripted_selection(self.measure, obs)))
