import math
from pathlib import Path

from wave_rl.env.env import EpisodeStatistics, build_env
from wave_rl.env.pools import load_pool
from wave_rl.video import BackgroundMontageRenderer, EpisodeVideoCapture


def train(ctx, logger) -> str | None:
    """Train the controller on ``<= n_train_max`` objects and return the path to its saved
    head (the eval sweep reloads it).

    **Training is never interrupted to measure.** The per-object-count curve comes from the
    training episodes themselves — :class:`wave_rl.env.env.EpisodeStatistics` buckets finished
    episodes by their own object count and this loop logs them as ``ep/*@Nobj`` — and the
    montage video is rendered by a separate process from captured joint states. The learning
    itself lives in the algorithm's ``learn`` / ``save``; everything here is harness wiring.
    """
    task = ctx.task
    n_max = task.train.n_train_max
    scene = ctx.scene(n_max)
    print(f"-- training '{task.algorithm}' on <= {n_max} objects "
          f"(budget={task.train.budget:,} "
          f"steps, {task.train.num_scenes} parallel scenes)")
    configs_q, _ = load_pool(task.train.pool_dir, task.scene, n_max, task.train.seed,
                             task.train.min_separation, task.train.anchor_stride,
                             task.train.n_configs)
    env = build_env(
        scene=scene, plant_cfg=task.plant, goal_cfg=ctx.goal,
        configs_q=configs_q, num_scenes=task.train.num_scenes,
        count_mode="mixed",
        episode_time_limit=task.train.time_limit,
        time_per_object=task.train.time_per_object,
        remove_on_completion=task.train.remove_on_completion,
        seed=task.train.seed)
    if task.train.active_count is not None:
        env.schedule.set_override(task.train.active_count)
        print(f"-- every training episode holds {task.train.active_count} objects")
    algo = ctx.build_algorithm(env)
    budget = task.train.budget
    algo.budget = budget
    if task.train.init_from is not None:
        algo.warm_start(task.train.init_from)
        print(f"== WARM START from {task.train.init_from} at "
              f"{algo.total_steps:,} steps — the head, its pins and its action space come "
              f"from the checkpoint, the optimiser moments do not and restart at zero, and "
              f"the learning rates run their schedule afresh over the steps this run adds. "
              f"budget={budget:,} is the TOTAL, so this run adds "
              f"{budget - algo.total_steps:,}.")
    algo.logger = logger

    renderer = _arm_video(task, ctx, env, scene, n_max)
    env.episode_stats = EpisodeStatistics(env.num_scenes, env.max_objects, env.device,
                                          task.train.stats_window_per_count)
    print(f"-- per-count training metrics: 'ep/*@Nobj' over a rolling "
          f"{task.train.stats_window_per_count} episodes per object count, logged every "
          f"{task.train.log_freq:,} steps. These are the EXPLORING policy's "
          f"numbers; the greedy held-out curve is the final sweep.")

    block = task.train.log_freq
    save_path = task.train.save_path or f"data/runs/{ctx.run_name}.pt"
    next_video = task.train.video_freq
    next_save = task.train.checkpoint_freq
    while algo.total_steps < budget:
        prev = algo.total_steps
        algo.learn(min(block, budget - algo.total_steps))
        if algo.total_steps == prev:
            print("-- algorithm.learn made no progress (non-learning); ending the loop")
            break
        step = algo.total_steps
        stats = env.episode_stats.metrics()
        record = logger.flush(step, {**stats, **env.plant.sim.drain_constraint_peak()})
        curve = " ".join(
            f"{c}obj={stats[f'ep/full_success_rate@{c}obj']:.2f}"
            for c in sorted(int(k.split("@")[1].split("obj")[0]) for k in stats
                            if k.startswith("ep/full_success_rate@")))
        print(f"-- t={step:,}  {record['time/env_steps_per_second']:,.0f} steps/s  "
              f"{record['time/elapsed_hours']:.2f} h  {curve or '(no finished episodes yet)'}")

        if renderer is not None:
            if step >= next_video:
                next_video = step + task.train.video_freq
                renderer.submit(env.video_capture.take(), step)
            for video in renderer.collect():
                # logged at the current step: wandb ignores a row whose step goes backwards
                logger.video("train_video/montage", video["path"], step)
                print(f"-- montage of the episodes captured at {video['step']:,} "
                      f"(object counts {video['counts']}) logged at {step:,}")

        if step >= next_save:
            next_save = step + task.train.checkpoint_freq
            algo.save(save_path)

        # Tested on every logging block, never nested inside the checkpoint gate: the
        # two cadences are unrelated.
        every = task.train.snapshot_every_steps
        if every > 0 and (step // every) > (prev // every):
            directory = Path("data/runs/snapshots") / ctx.run_name
            directory.mkdir(parents=True, exist_ok=True)
            retained = directory / f"step_{step:012d}.pt"
            algo.save(str(retained))
            print(f"-- retained snapshot: {retained}")
    if renderer is not None:
        renderer.close()
    if algo.total_steps == 0:
        print("-- zero-shot algorithm: no parameters, so no checkpoint is written and "
              "the sweep runs the algorithm it builds itself")
        return None
    algo.save(save_path)
    print(f"-- trained head saved to {save_path}")
    return save_path


def _arm_video(task, ctx, env, scene, n_max: int):
    """Capture a few training scenes' joint states and render them OUT OF PROCESS: rendering
    is CPU/GL work that would otherwise block the training loop, so the parent only writes a
    few MB of joint states."""
    if not (task.train.video and task.train.video_freq > 0):
        return None
    scenes = list(range(min(task.train.video_episodes, env.num_scenes)))
    seconds = (task.train.time_per_object * n_max
               if task.train.time_per_object is not None else task.train.time_limit)
    max_ticks = math.ceil(seconds / task.plant.tau_step) + 5
    env.video_capture = EpisodeVideoCapture(scenes, n_max, max_ticks, env.device,
                                            stride=task.train.video_stride)
    renderer = BackgroundMontageRenderer(
        str(Path("data/runs/montage") / ctx.run_name), scene,
        fps=max(5, 50 // task.train.video_stride))
    print(f"-- montage video: {len(scenes)} training scenes captured every "
          f"{task.train.video_stride} ticks and rendered OUT OF PROCESS every "
          f"{task.train.video_freq:,} steps -- training never waits for it")
    return renderer
