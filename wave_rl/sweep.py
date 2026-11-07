import csv
import math
import time
from pathlib import Path

import torch

from wave_rl.env.env import build_env
from wave_rl.env.pools import load_pool


#: why an episode ended, by code. `object_lost` and `sim_failure` are the plant killing the
#: episode, not the clock: folding them into `timeout` makes a dead scene read as a slow
#: policy.
END_REASONS = ("timeout", "success", "aborted", "rollout_cap", "object_lost", "sim_failure")


def run_episodes(vec_env, algo, max_steps: int) -> dict:
    """Roll the greedy policy on every scene in parallel until each has finished ONE episode,
    or ``max_steps``, and summarise those episodes.

    One episode per scene and never more. Successes end early and failures run to the clock,
    so the extra episodes a fast scene finishes while a slow one is still running are mostly
    successes, and counting them would inflate every rate."""
    obs = vec_env.reset()
    algo.reset()
    B = vec_env.num_scenes
    dev = obs.active_mask.device
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    objects = torch.zeros(B, dtype=torch.long, device=dev)
    completed = torch.zeros_like(objects)
    steps = torch.full_like(objects, max_steps)
    reason = torch.full_like(objects, END_REASONS.index("rollout_cap"))
    code = {name: torch.full_like(objects, k) for k, name in enumerate(END_REASONS)}

    placed = torch.zeros_like(obs.active_mask)
    ever_placed = torch.zeros_like(obs.active_mask)
    knocks = torch.zeros_like(objects)

    dones = torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(1, max_steps + 1):
        action = algo.act_batched(obs, dones)
        obs, obs_true, terminated, truncated, info = vec_env.step(action)
        dones = terminated | truncated
        newly = dones & ~finished
        n_objects = obs_true.active_mask.sum(dim=1)
        now_placed = obs_true.satisfied_mask & obs_true.active_mask
        n_done = now_placed.sum(dim=1)
        live = ~finished
        knocks = knocks + ((placed & ~now_placed).sum(dim=1) * live)
        ever_placed |= now_placed & live[:, None]
        placed = now_placed
        why = torch.where(truncated, code["timeout"],
                          torch.where(n_done >= n_objects, code["success"], code["aborted"]))
        why = torch.where(info["unachievable"], code["object_lost"], why)
        why = torch.where(info["sim_failure"], code["sim_failure"], why)
        objects = torch.where(newly, n_objects, objects)
        completed = torch.where(newly, n_done, completed)
        steps = torch.where(newly, torch.full_like(steps, t), steps)
        reason = torch.where(newly, why, reason)
        finished |= dones
        if bool(finished.all()):
            break

    still = ~finished
    objects = torch.where(still, obs.active_mask.sum(dim=1), objects)
    completed = torch.where(still, (obs.satisfied_mask & obs.active_mask).sum(dim=1), completed)
    return {
        "full_success_rate": float((completed == objects).float().mean()),
        "mean_n_completed": float(completed.float().mean()),
        "mean_steps": float(steps.float().mean()),
        # a placed object that stops being placed: knocked off, or a support pulled from under it
        "mean_knocks": float(knocks.float().mean()),
        # the objects in place at the end over the objects that were ever in place
        "placed_retention": float(completed.sum() / ever_placed.sum().clamp(min=1)),
        **{f"episode_end_{name}_fraction": float((reason == k).float().mean())
           for k, name in enumerate(END_REASONS)},
    }


def resolve_eval_pool(task, counts: list[int]) -> tuple:
    """The layout pool this evaluation draws from, ANNOUNCED: which layouts a number was
    measured on is stated rather than assumed.

    The sweep rolls every count on ONE ``max(counts)``-object scene, so the pool is that
    scene's. ``eval.configs_seed`` = null draws it with the training seed — at
    ``max(counts) == train.n_train_max`` the very layouts the policy trained on; any other
    seed is a held-out draw of the same distribution."""
    ev, tr = task.eval, task.train
    n = max(counts)
    pool_seed = tr.seed if ev.configs_seed is None else ev.configs_seed
    configs_q, path = load_pool(tr.pool_dir, task.scene, n, pool_seed, tr.min_separation,
                                tr.anchor_stride, tr.n_configs)
    is_train = pool_seed == tr.seed and n == tr.n_train_max
    ood = [c for c in counts if c > tr.n_train_max]
    print("=" * 78)
    print(f"== EVAL LAYOUT POOL: {path} (seed {pool_seed}, {len(configs_q)} rows)")
    print(f"==   goal             : {task.goal.type} tower_height={task.goal.tower_height} "
          f"stack_dz={task.goal.stack_dz} eps_pos={task.goal.eps_pos} "
          f"success_mode={task.goal.success_mode}")
    print("==   VERDICT          : " + ("*** TRAINING POOL *** — these layouts were seen "
                                         "during training" if is_train else
                                         "not the pool the policy trained on"))
    print(f"==   n_train_max      : {tr.n_train_max}"
          + (f"   -> OOD counts in this sweep: {ood}" if ood else "   (no OOD counts)"))
    print("=" * 78)
    return configs_q, {"pool_path": str(path), "pool_seed": pool_seed,
                       "pool_is_training_pool": is_train}


def final_sweep(ctx, counts: list[int], logger) -> None:
    """The run's ONLY held-out, greedy-policy measurement, as per-count BATCHED rollouts in
    one process.

    ``eval.n_episodes`` scenes are rolled per count in parallel on ONE vectorized env, through
    the count-override mechanism, rather than forking per-count GPU workers that
    context-thrash a single device. Training itself is measured continuously under ``ep/*``,
    but those are the EXPLORING policy's numbers on the TRAINING pool; these are not.

    Counts whose goal structure does not fit the arena are dropped; every count reports its
    constraint-row health, so a starved plant reads as starved rather than as a
    generalization cliff.
    """
    task = ctx.task
    n_eps = task.eval.n_episodes
    tpo = task.eval.time_per_object

    n_max = max(counts)
    scene = ctx.scene(n_max)
    configs, provenance = resolve_eval_pool(task, counts)
    env = build_env(
        scene=scene, plant_cfg=task.plant, goal_cfg=ctx.goal,
        configs_q=configs, num_scenes=n_eps, count_mode="fixed",
        episode_time_limit=task.eval.time_limit, time_per_object=tpo,
        remove_on_completion=not task.eval.keep_completed,
        seed=task.eval.seed + 13)
    for key, value in task.eval.algorithm_overrides.items():
        print(f"-- eval.algorithm_overrides: {key} = {value!r}")
    algo = ctx.build_algorithm(env, task.eval.algorithm_overrides)
    if ctx.trained_path is not None:
        algo.load(ctx.trained_path)
    unfit = [c for c in counts if not env.task_goal.fits(c)]
    if unfit:
        print(f"!! final sweep: the goal layout does not fit the reachable arena "
              f"(|xy| <= {env.task_goal.placeable_xy:.3f} m) at counts {unfit}; SKIPPING them")
        counts = [c for c in counts if c not in unfit]
    print(f"-- BATCHED final sweep: counts {counts}, {n_eps} parallel episodes per count on "
          f"ONE {n_max}-object scene (surplus bodies parked)")

    tau = task.plant.tau_step
    rows = []
    for c in counts:
        env.schedule.set_override(c)
        max_steps = math.ceil((tpo * c if tpo is not None else task.eval.time_limit) / tau) + 5
        started = time.time()
        stats = run_episodes(env, algo, max_steps)
        metrics = {
            **{f"sweep/{k}": v for k, v in stats.items()},
            "sweep/eval_pool_is_training_pool": float(provenance["pool_is_training_pool"]),
            "sweep/eval_pool_seed": float(provenance["pool_seed"]),
            "sweep/scene_objects": float(len(scene.obj_names)),
            "sweep/sweep_seconds": float(time.time() - started),
        }
        metrics.update({f"sweep/{k.removeprefix('simulator/')}": v
                        for k, v in env.plant.sim.drain_constraint_peak().items()})
        logger.sweep(c, metrics)
        rows.append({"objects": c, **metrics})
        ends = "/".join(f"{name} {stats[f'episode_end_{name}_fraction']:.2f}"
                        for name in END_REASONS)
        print(f"  [{c} obj] full_success={stats['full_success_rate']:.2f} "
              f"mean_completed={stats['mean_n_completed']:.2f} "
              f"knocks={stats['mean_knocks']:.2f} retention={stats['placed_retention']:.2f} "
              f"end[{ends}] "
              f"[{time.time() - started:.0f}s, scene={len(scene.obj_names)}obj, "
              f"pool={'TRAIN' if provenance['pool_is_training_pool'] else 'held-out'}]")
    out = Path("data/runs") / f"{ctx.run_name}_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        table = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
                               restval="")
        table.writeheader()
        table.writerows(rows)
    print(f"-- sweep written to {out}")
