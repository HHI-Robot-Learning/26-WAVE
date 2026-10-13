import dataclasses

import torch

from wave_rl.logger import Logger
from wave_rl.seeding import set_global_seeds
from wave_rl.sweep import final_sweep
from wave_rl.task import resolve_run
from wave_rl.training import train


def run(task_name: str, overrides: list[str]) -> None:
    """Run one task end to end: train the controller if ``train.enabled``, then sweep the
    object counts if ``eval.enabled``. With training off, a learned algorithm loads the head at
    ``train.save_path`` and goes straight to the sweep, the path a zero-shot baseline takes."""
    # TF32 matmuls: a free speedup for the matmul-bound heads; the simulator is unaffected.
    torch.set_float32_matmul_precision("high")
    ctx, counts = resolve_run(task_name, overrides)
    task = ctx.task
    set_global_seeds(task.train.seed)

    # One record stream for the whole task: `ep/*` against the training step, `sweep/*`
    # against the object count.
    logger = Logger(ctx.run_name, "cuda", {"project": "wave", "group": task_name}
                    if task.wandb else None)
    logger.config({"task": task_name, **dataclasses.asdict(task),
                   "goal": dataclasses.asdict(ctx.goal), "eval_counts": counts})

    if task.train.enabled:
        ctx.trained_path = train(ctx, logger)
    elif task.train.save_path is not None:
        ctx.trained_path = task.train.save_path
        print(f"-- eval-only: loading the trained head from {ctx.trained_path}")
    if task.eval.enabled:
        final_sweep(ctx, counts, logger)
    logger.finish()
