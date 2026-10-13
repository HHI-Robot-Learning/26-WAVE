import dataclasses

import omegaconf
import torch

from wave_rl.algorithms import ALGORITHMS
from wave_rl.algorithms.lifter import Lifter
from wave_rl.config import CONFIGS, GoalConfig, TaskConfig
from wave_rl.sim.scene import multi_object_scene
from wave_rl.skills.embodiments import load_skill
from wave_rl.skills.td7 import load_skill_config

def task_file(task: str) -> omegaconf.DictConfig:
    """``configs/tasks/<task>.yaml``, merged over the task it names as ``base`` (if any)."""
    raw = omegaconf.OmegaConf.load(CONFIGS / "tasks" / f"{task}.yaml")
    if "base" not in raw:
        return raw
    base = raw.pop("base")
    return omegaconf.OmegaConf.merge(task_file(base), raw)


def load_task(task: str, overrides: list[str]) -> TaskConfig:
    """A task file with its plant loaded from ``configs/plant/<plant>.yaml`` and
    ``overrides`` (``key.sub=value``) on top, loaded onto :class:`TaskConfig` so values are
    type-checked and unknown keys fail at startup."""
    raw = task_file(task)
    raw.plant = omegaconf.OmegaConf.load(CONFIGS / "plant" / f"{raw.plant}.yaml")
    cfg = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.structured(TaskConfig), raw,
                                    omegaconf.OmegaConf.from_dotlist(overrides))
    return omegaconf.OmegaConf.to_object(cfg)


def resolve_goal(goal: GoalConfig, skill: str | None) -> GoalConfig:
    """Take the goal tolerance from the frozen skill's sidecar where the task leaves it unset.
    With no skill there is nothing to inherit from, so the task must state it."""
    if goal.eps_pos is not None:
        return goal
    if skill is None:
        raise SystemExit("goal.eps_pos must be set for an algorithm with no frozen skill.")
    return dataclasses.replace(goal, eps_pos=load_skill_config(skill).eps_pos)


def inherit_checkpoint_config(task: TaskConfig, checkpoint_path: str) -> None:
    """EVAL-ONLY REPLAY: a WAVE-family head is replayed under the algorithm config it was
    TRAINED with, read from the checkpoint.

    A head is trained under one projection, one measure and one decision clock; replayed under
    different values it scores confidently and wrongly. A head that records no config (an
    end-to-end baseline) keeps the task's. Recorded keys the config does not have are
    settings the code holds fixed."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "warp_config" not in checkpoint:
        return
    print(f"== EVAL-ONLY REPLAY: algorithm config from {checkpoint_path}")
    _, config_cls = ALGORITHMS[task.algorithm]
    fields = {f.name for f in dataclasses.fields(config_cls)}
    task.algorithm_cfg = {k: v for k, v in checkpoint["warp_config"].items() if k in fields}
    if checkpoint["frozen_agent"] != task.skill:
        raise SystemExit(f"{checkpoint_path} was trained on the skill "
                         f"{checkpoint['frozen_agent']!r}, the task names {task.skill!r}. A head "
                         f"replayed over a different skill scores confidently and wrongly.")


@dataclasses.dataclass
class RunContext:
    """Everything resolved once per run, count-independent."""

    task: TaskConfig
    run_name: str
    goal: GoalConfig                # resolved: concrete eps_pos
    trained_path: str | None = None   # learned head to load at eval (None = zero-shot)

    def scene(self, n_objects: int):
        p = self.task.plant
        return multi_object_scene(self.task.scene, n_objects, p.friction, p.solref, p.solimp)

    def build_algorithm(self, env, overrides: dict | None = None):
        """Instantiate the configured algorithm against ``env``, with ``overrides`` on top of
        its config. A :class:`Lifter` is handed the frozen skill, sized to THIS env's robot."""
        cls, config_cls = ALGORITHMS[self.task.algorithm]
        cfg = omegaconf.OmegaConf.to_object(omegaconf.OmegaConf.merge(
            omegaconf.OmegaConf.structured(config_cls), self.task.algorithm_cfg,
            overrides or {}))
        if issubclass(cls, Lifter):
            skill = load_skill(self.task.skill, env.scene, env.n_robot_qpos, env.n_robot_qvel,
                               env.n_ctrl, env.action_dim, env.device)
            env.attach_kinematics(skill)
            return cls(skill, env, cfg)
        return cls(env, cfg)


def resolve_run(task_name: str, overrides: list[str]) -> tuple[RunContext, list[int]]:
    """Resolve one run: the task, its goal tolerance and its eval count schedule."""
    task = load_task(task_name, overrides)
    if not task.train.enabled and task.train.save_path is not None:
        inherit_checkpoint_config(task, task.train.save_path)
    print(f"Task {task_name!r} [{task.algorithm}] on {task.scene}: "
          + (f"frozen skill {task.skill}" if task.skill else "no frozen skill"))
    goal = resolve_goal(task.goal, task.skill)
    counts = list(range(task.eval.start, task.eval.stop + 1, task.eval.step))
    print(f"Eval object counts: {counts}")
    run_name = task.run_name or f"{task_name}_seed{task.train.seed}"
    return RunContext(task=task, run_name=run_name, goal=goal), counts
