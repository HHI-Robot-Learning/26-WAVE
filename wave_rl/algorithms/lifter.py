from wave_rl.algorithms.base import Algorithm
from wave_rl.skills.base import FrozenSkill


class Lifter(Algorithm):
    """An algorithm that drives a frozen single-object skill over a multi-object scene.

    It owns exactly one thing the skill does not: which object to work, and — for a learned
    lifter — how to bend the skill's inputs and outputs while it does. The skill owns its
    own observation, its localisation and its execution, so a lifter never touches an
    embodiment constant and works unchanged on every robot the skill package supports.
    """

    def __init__(self, skill: FrozenSkill, env, cfg) -> None:
        super().__init__(env=env, cfg=cfg)
        self.skill = skill
        self.layout = skill.layout

    def learn(self, n_steps: int) -> None:
        """No-op: a heuristic lifter is zero-shot. A learned lifter overrides this."""
