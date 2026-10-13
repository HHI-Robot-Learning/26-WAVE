import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp

from wave_rl.algorithms.wave.algorithm import WaveAlgorithm
from wave_rl.algorithms.wave.runtime import WaveRuntime
from wave_rl.env.env import build_env
from wave_rl.sweep import resolve_eval_pool
from wave_rl.task import resolve_run
from wave_rl.video import render_payload

#: what a world executes. The deployed world runs the episode; every other world is a copy
#: that branches off its state at every tick and runs for `horizon` ticks.
ARMS = ("deployed", "warper off", "input warp off", "output warp off", "deployed again",
        "warper off pulse")
MAIN, OFF, IN_OFF, OUT_OFF, AGAIN, PULSE_OFF = range(6)
BRANCH_ARMS = (OFF, IN_OFF, OUT_OFF, PULSE_OFF, AGAIN)
#: per-tick records of the deployed world, stacked to (ticks, ...) per episode
RECORDED = ("phase", "n_satisfied", "skill_action", "executed_action", "input_warped_action",
            "output_warped_action", "branch_ticks", "qpos", "goals_qpos", "active")

PHASES = ("approach", "grasp", "carry", "release & depart")
APPROACH, GRASP, CARRY, RELEASE = range(4)
PHASE_COLORS = {"approach": "#4C72B0", "grasp": "#DD8452", "carry": "#55A868",
                "release & depart": "#C44E52"}
IN_C, OUT_C = "#8172B3", "#DD8452"

#: Per-world state, by owner: every tensor a rollout carries from one control tick to the
#: next and that differs between worlds. `(name, world_axis)`. These are SAVED, RESTORED and
#: COPIED FROM ONE WORLD INTO OTHERS. Everything scalar or 0-dim on the same owners is saved
#: and restored generically; every dict of tensors likewise.
PER_WORLD = {
    "env": [(n, 0) for n in (
        "active_count", "nominal_goals_feat", "goals_feat", "goals_qpos", "completed",
        "last_status", "_sat_streak", "episode_time", "_last_q_live",
        "_nan_world_mask", "_prev_ee")],
    "plant": [(n, 0) for n in (
        "ctrl_current", "grip_closed")]
    + [("_ctrl_buffer", 1)],
    "sim": [("_qpos_t", 0), ("_qvel_t", 0), ("_act_t", 0), ("_xpos_t", 0)],
    "solver": [("qacc_warmstart", 0)],
    "algo": [("_greedy_warp", 0), ("_greedy_prev", 0), ("_pending_done", 0)],
    "runtime": [(n, 0) for n in (
        "rest_z", "atom", "has_atom", "atom_cooldown", "atom_ever_sat", "prev_sat",
        "tick_in_episode")],
}


class BranchableState:
    """Every piece of state the rollout carries — env bookkeeping, the plant's actuation
    latches, the physics, the solver warm-start, the warper's committed decision and the
    runtime's selection state — as one thing that can be saved, restored, and copied from one
    world into a block of others. A physics step reads only the positions, velocities, the
    reference it follows and the solver's warm start, so a world restored from these continues
    exactly as the original would have."""

    def __init__(self, env, algo) -> None:
        sim = env.plant.sim
        self.env, self.algo = env, algo
        self.B = env.num_scenes
        self.solver = dataclasses.make_dataclass("Solver", ["qacc_warmstart"])(
            wp.to_torch(sim.data.qacc_warmstart))

    def owners(self) -> dict:
        env = self.env
        return {"env": env, "plant": env.plant, "sim": env.plant.sim, "solver": self.solver,
                "algo": self.algo, "runtime": self.algo._state}

    def _generic(self, obj) -> dict:
        """The scalars, 0-dim tensors and dicts of tensors on one owner."""
        out = {}
        for name, value in vars(obj).items():
            if isinstance(value, (bool, int, float)) or value is None:
                out[name] = value
            elif torch.is_tensor(value) and value.dim() == 0:
                out[name] = value.clone()
            elif (isinstance(value, dict) and value
                  and all(torch.is_tensor(v) for v in value.values())):
                out[name] = {k: v.clone() for k, v in value.items()}
        return out

    def save(self) -> dict:
        snap = {}
        for owner, obj in self.owners().items():
            if obj is None:
                snap[owner] = None
                continue
            tensors = {}
            for name, _axis in PER_WORLD[owner]:
                value = getattr(obj, name)
                tensors[name] = None if value is None else value.clone()
            snap[owner] = {"tensors": tensors, "generic": self._generic(obj)}
        return snap

    def restore(self, snap: dict) -> None:
        if snap["runtime"] is None:
            self.algo._state = None
        for owner, obj in self.owners().items():
            if snap[owner] is None:
                continue
            for name, saved in snap[owner]["tensors"].items():
                live = getattr(obj, name)
                if saved is None:
                    setattr(obj, name, None)
                elif (torch.is_tensor(live) and live.shape == saved.shape
                      and live.dtype == saved.dtype):
                    # in place: several of these are views of simulator memory
                    live.copy_(saved)
                else:
                    setattr(obj, name, saved.clone())
            for name, saved in snap[owner]["generic"].items():
                if torch.is_tensor(saved):
                    setattr(obj, name, saved.clone())
                elif isinstance(saved, dict):
                    setattr(obj, name, {k: v.clone() for k, v in saved.items()})
                else:
                    setattr(obj, name, saved)

    def replicate(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """World `src[i]` -> every world in `dst[i]`, for every per-world tensor and every
        per-world value inside a dict."""
        s = src.unsqueeze(1).expand_as(dst).reshape(-1)
        d = dst.reshape(-1)
        for owner, obj in self.owners().items():
            if obj is None:
                continue
            for name, axis in PER_WORLD[owner]:
                value = getattr(obj, name)
                if value is None:
                    continue
                if axis == 0:
                    value[d] = value[s]
                else:
                    value[:, d] = value[:, s]
            for value in vars(obj).values():
                if isinstance(value, dict):
                    for v in value.values():
                        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == self.B:
                            v[d] = v[s]

class Controller:
    """The deployed controller, tick for tick, from `WaveAlgorithm`'s own pieces, so the skill's
    own action, the action under each side of the warp and under both are all visible, and every
    world executes its own arm while the warper keeps deciding."""

    def __init__(self, algo: WaveAlgorithm) -> None:
        self.algo = algo

    @torch.no_grad()
    def tick(self, obs, arm: torch.Tensor) -> dict:
        algo = self.algo
        if algo._state is None:
            WaveRuntime.act_batched(algo, obs)
        algo._decide_warp(obs)
        algo._pending_done = None
        naive = WaveRuntime.act_batched(algo, obs)
        if algo._greedy_identity:
            return {"action": naive, "naive": naive, "under_input": naive, "under_output": naive}
        w = algo._greedy_warp
        live = algo._state.has_atom.unsqueeze(1)
        under_input = torch.where(live, algo._warped_input_action(obs, w), naive)
        under_output = torch.where(live, algo._warp_output(naive, obs, w), naive)
        both = torch.where(live, algo._warp_output(under_input, obs, w), naive)
        a = arm.unsqueeze(1)
        action = torch.where(a == OFF, naive,
                             torch.where(a == IN_OFF, under_output,
                                         torch.where(a == OUT_OFF, under_input, both)))
        return {"action": action, "naive": naive, "under_input": under_input,
                "under_output": under_output}


def clone_obs(obs):
    return dataclasses.replace(obs, **{
        f.name: getattr(obs, f.name).clone() for f in dataclasses.fields(obs)
        if getattr(obs, f.name) is not None})


def replicate_obs(obs, src: torch.Tensor, dst: torch.Tensor):
    out = clone_obs(obs)
    s = src.unsqueeze(1).expand_as(dst).reshape(-1)
    d = dst.reshape(-1)
    for f in dataclasses.fields(out):
        value = getattr(out, f.name)
        if value is not None:
            value[d] = value[s]
    return out


def plant_state(obs, algo, goal, rest_z: torch.Tensor, under: float) -> dict:
    """The plant facts every phase is judged on, per world: hand-to-object xy distance in
    object heights, which objects are carried (the carry test of `carried_flag`, applied to
    every object: lifted and under the hand), which are over their goal (inside the goal's own
    xy acceptance radius, from the placed height up to one object height above it: where a held
    object is let go) and which are placed."""
    feat, target = obs.obj_feat.float(), obs.goal_feat.float()
    dist = torch.linalg.norm(feat[..., :2] - obs.ee_position[:, None, :2].float(), dim=-1) / algo.h
    dist = torch.where(obs.active_mask, dist, torch.full_like(dist, float("inf")))
    supported = algo.measure.supported(obs, rest_z)
    dz = feat[..., 2] - target[..., 2]
    over = ((torch.linalg.norm(feat[..., :2] - target[..., :2], dim=-1) <= goal.xy_eps)
            & (dz >= -goal.eps_pos) & (dz <= goal.eps_pos + algo.h))
    return {"dist": dist, "carried": obs.active_mask & ~supported & (dist < under),
            "over_goal": over & obs.active_mask, "sat": obs.satisfied_mask & obs.active_mask}


def step_done(phase: torch.Tensor, obj: torch.Tensor, plant: dict, closed: torch.Tensor,
              at: float, under: float) -> torch.Tensor:
    """(B,) each world's current phase is finished on its object.
    approach          the hand is at the object (aligned within `at` object heights)
    grasp             the object is lifted in closed jaws
    carry             the object is over its goal, or placed
    release & depart  the object is placed and the hand has left it (beyond `under`): it was let
                      go, since a held object leaves with the hand"""
    rows = torch.arange(phase.shape[0], device=phase.device)
    dist = plant["dist"][rows, obj]
    carried = plant["carried"][rows, obj]
    sat = plant["sat"][rows, obj]
    over = plant["over_goal"][rows, obj]
    return torch.where(phase == APPROACH, dist < at,
                       torch.where(phase == GRASP, carried & closed,
                                   torch.where(phase == CARRY, over | sat,
                                               (dist >= under) & sat)))


def advance_phase(machine: dict, closed: bool, atom: int, dist, carried, over_goal,
                  sat, at: float, under: float) -> int:
    """One tick of the deployed episode's plant state -> its phase. The phases of one object
    only move forward, approach -> grasp -> carry -> release & depart, and step back only on a
    real failure: the object dropped away from the hand, or let go without ending up placed.

    approach          until the hand is at the object
    grasp             until the object is lifted in closed jaws
    carry             until the object is over its goal or placed, or the jaws open over it
    release & depart  until the hand has left it; placed -> the next object's approach"""
    o, m = machine["object"], machine["phase"]
    if o is None or (m in (APPROACH, GRASP) and o != atom):
        o, m = atom, APPROACH
    if carried[o] and closed:
        m = RELEASE if (sat[o] or over_goal[o]) else CARRY
    elif m == CARRY:
        if sat[o] or over_goal[o] or not closed:
            m = RELEASE
        elif dist[o] >= under:
            m = APPROACH
    elif m == RELEASE and dist[o] >= under:
        if sat[o] and atom != o:
            o, m = atom, APPROACH
        elif not sat[o]:
            m = APPROACH
    if m == APPROACH and dist[o] < at:
        m = GRASP
    machine["object"], machine["phase"] = o, m
    return m


def phase_ticks(env, ctrl, algo, obs, arm, phase, obj, horizon: int, pulse: int,
                at: float, under: float) -> torch.Tensor:
    """(B,) ticks each world takes to finish its current phase on its object, from the state
    it holds now, every world on its own arm. A world that does not finish within the horizon,
    un-places a placed object, or has its episode cut short counts the whole horizon."""
    B, dev = env.num_scenes, env.device
    sat0 = obs.satisfied_mask & obs.active_mask
    done_at = torch.zeros(B, device=dev)
    damage = torch.zeros(B, dtype=torch.bool, device=dev)
    over = torch.zeros(B, dtype=torch.bool, device=dev)
    for k in range(1, horizon + 1):
        now = torch.where(arm == PULSE_OFF, torch.full_like(arm, OFF if k <= pulse else MAIN),
                          arm)
        obs, true, terminated, truncated, info = env.step(ctrl.tick(obs, now)["action"])
        plant = plant_state(true, algo, env.task_goal, algo._state.rest_z, under)
        finished = step_done(phase, obj, plant, env.plant.grip_closed.bool(), at, under) | (
            terminated & ~info["sim_failure"])
        done_at = torch.where((done_at == 0) & finished & ~over, float(k), done_at)
        damage |= ~over & ((sat0 & ~plant["sat"]).any(1) | truncated | info["sim_failure"])
        over |= terminated | truncated
    return torch.where((done_at == 0) | damage, float(horizon), done_at)


def ordered(phase: np.ndarray, n_satisfied: np.ndarray, success: bool, n: int) -> bool:
    """The episode built its structure in one pass: it succeeded, no placement was ever lost
    and every object was carried exactly once. The jaws may chatter -- only a second carry of
    the same object is a retry."""
    carries = int(np.sum((phase[1:] == CARRY) & (phase[:-1] != CARRY) & (phase[:-1] != RELEASE))
                  + (phase[0] == CARRY))
    return bool(success and not np.any(np.diff(n_satisfied) < 0) and carries == n)


def record(args) -> None:
    """Deployed episodes recorded tick by tick, with branched copies on every arm from every
    tick -> out_dir/necessity.json, necessity.npz and the rendered scene views."""
    torch.set_float32_matmul_precision("high")
    ctx, _ = resolve_run(args.task, [*args.overrides, "train.enabled=false",
                                     f"train.save_path={args.checkpoint}",
                                     f"eval.configs_seed={args.pool_seed}"])
    ctx.trained_path = args.checkpoint
    task = ctx.task
    task_name = args.task
    os.makedirs(args.out_dir, exist_ok=True)

    n, E, H, M = args.n_objects, args.episodes, args.horizon, args.replicas
    pulse = args.pulse_ticks
    branch_arms = [a for a in BRANCH_ARMS for _ in range(M)]
    R = 1 + len(branch_arms)
    B = E * R
    scene = ctx.scene(n)
    configs, provenance = resolve_eval_pool(task, [n])
    tpo = float(task.eval.time_per_object) if task.eval.time_per_object is not None else None
    env = build_env(
        scene=scene, plant_cfg=task.plant, goal_cfg=ctx.goal,
        configs_q=configs, num_scenes=B, count_mode="fixed",
        episode_time_limit=float(task.eval.time_limit), time_per_object=tpo,
        remove_on_completion=not bool(task.eval.keep_completed),
        seed=int(task.eval.seed) + 13)
    algo = ctx.build_algorithm(env)
    if not isinstance(algo, WaveAlgorithm):
        raise SystemExit(f"warp_necessity measures a WAVE head; task {task_name!r} builds "
                         f"{type(algo).__name__}")
    algo.load(args.checkpoint)
    dev = env.device
    tau = env.tau_step

    print("=" * 78)
    print(f"== WARP NECESSITY  task={task_name} objects={n} checkpoint={args.checkpoint}")
    print(f"==   {E} episodes x ({M} copies x {[ARMS[a] for a in BRANCH_ARMS]}) = {B} worlds; "
          f"copies from every tick, {H} ticks = {H * tau:.1f} s long; the pulse arm is off for "
          f"its first {pulse} ticks, every other arm throughout")
    print("=" * 78)

    ctrl = Controller(algo)
    state = BranchableState(env, algo)
    obs = env.reset()
    algo.reset()
    deployed = torch.full((B,), MAIN, dtype=torch.long, device=dev)

    arm_t = torch.tensor([MAIN, *branch_arms], device=dev).repeat(E)
    group_t = torch.arange(B, device=dev) // R
    main_w = torch.arange(E, device=dev) * R
    replicas = main_w.unsqueeze(1) + torch.arange(1, R, device=dev).unsqueeze(0)
    alive = torch.ones(E, dtype=torch.bool, device=dev)
    rec = [{k: [] for k in RECORDED} for _ in range(E)]
    success = [False] * E
    machines = [{"object": None, "phase": APPROACH} for _ in range(E)]
    started = time.time()
    tick = 0
    while bool(alive.any()):
        alive_np = alive.cpu().numpy()
        branch = torch.zeros(E, R - 1, device=dev)
        if tick > 0:
            phase = torch.zeros(B, dtype=torch.long, device=dev)
            obj = torch.zeros(B, dtype=torch.long, device=dev)
            for e in np.nonzero(alive_np)[0]:
                phase[group_t == e] = int(rec[e]["phase"][-1])
                obj[group_t == e] = int(machines[e]["object"])
            snap = state.save()
            saved_obs = clone_obs(obs)
            state.replicate(main_w, replicas)
            branch = phase_ticks(env, ctrl, algo, replicate_obs(obs, main_w, replicas), arm_t,
                                 phase, obj, H, pulse, args.at, args.under)[replicas]
            state.restore(snap)
            obs = saved_obs

        out = ctrl.tick(obs, deployed)
        rt = algo._state
        atom = rt.atom[main_w].cpu().numpy()
        obs, true, terminated, truncated, info = env.step(out["action"])
        tensors = {"skill_action": out["naive"], "executed_action": out["action"],
                   "input_warped_action": out["under_input"],
                   "output_warped_action": out["under_output"], "qpos": env._last_q_live,
                   "goals_qpos": env.goals_qpos, "active": true.active_mask}
        tensors = {k: v[main_w].float().cpu().numpy() for k, v in tensors.items()}
        tensors["branch_ticks"] = branch.cpu().numpy()
        plant = {k: v[main_w].cpu().numpy()
                 for k, v in plant_state(true, algo, env.task_goal, rt.rest_z,
                                         args.under).items()}
        closed = env.plant.grip_closed[main_w].cpu().numpy()
        for e in np.nonzero(alive_np)[0]:
            rec[e]["phase"].append(advance_phase(
                machines[e], bool(closed[e]), int(atom[e]), plant["dist"][e],
                plant["carried"][e], plant["over_goal"][e], plant["sat"][e], args.at,
                args.under))
            rec[e]["n_satisfied"].append(int(plant["sat"][e].sum()))
            for key, value in tensors.items():
                rec[e][key].append(value[e])
        done = (terminated | truncated)[main_w] & alive
        won = (terminated & ~info["sim_failure"])[main_w].cpu().numpy()
        for e in np.nonzero(done.cpu().numpy())[0]:
            success[e] = bool(won[e])
        alive &= ~done
        tick += 1
        if tick % 10 == 0:
            print(f"   t={tick * tau:6.2f}s  {int(alive.sum())}/{E} episodes running  "
                  f"[{time.time() - started:.0f}s]")

    rec = [{k: np.stack(v) for k, v in r.items()} for r in rec]
    one_pass = [ordered(r["phase"], r["n_satisfied"], success[e], n) for e, r in enumerate(rec)]
    meta = {"task": task_name, "checkpoint": args.checkpoint, "n_objects": n, "episodes": E,
            "tick_seconds": tau, "replicas": M, "horizon": H, "pulse_ticks": pulse,
            "branch_arms": [ARMS[a] for a in branch_arms], "pool": provenance,
            "seconds": time.time() - started}
    with open(os.path.join(args.out_dir, "necessity.json"), "w") as fh:
        json.dump({"meta": meta, "episodes": [
            {"episode": e, "success": success[e], "one_pass": one_pass[e]} for e in range(E)]},
            fh)
    np.savez_compressed(os.path.join(args.out_dir, "necessity.npz"), **{
        f"ep{e:02d}_{key}": value for e, r in enumerate(rec) for key, value in r.items()})
    print(f"== WROTE {args.out_dir}/necessity.json and necessity.npz")
    # the scene views for the figure, joint states -> offscreen GL, done LAST so a render
    # failure cannot cost the measurements
    xml_path = Path(args.out_dir) / "scene.xml"
    xml_path.write_text(scene.xml)
    for e in [e for e in range(E) if one_pass[e]][:args.render_episodes]:
        r = rec[e]
        arrays = {"qpos_0": r["qpos"].astype(float)[::2], "goals_0": r["goals_qpos"][0].astype(float),
                  "active_0": r["active"][0] > 0.5}
        frames = render_payload({"xml_path": str(xml_path),
                                 "half_size": [*scene.object.half_xy, scene.object.half_z],
                                 "episodes": [{"scene": e, "count": n}]}, arrays)
        np.savez_compressed(os.path.join(args.out_dir, f"frames_ep{e:02d}.npz"),
                            frames=np.stack(frames).astype(np.uint8),
                            ticks=np.arange(0, len(r["qpos"]), 2))
        print(f"-- rendered episode {e}: {len(frames)} frames of {frames[0].shape}")


def load(out_dir: str) -> tuple[dict, list]:
    with open(os.path.join(out_dir, "necessity.json")) as fh:
        lab = json.load(fh)
    raw = np.load(os.path.join(out_dir, "necessity.npz"))
    return lab["meta"], [ep | {k: raw[f"ep{ep['episode']:02d}_{k}"] for k in RECORDED}
                         for ep in lab["episodes"]]


def cost(ep: dict, meta: dict, arm: int) -> np.ndarray:
    """(T,) mean over the arm's copies of the time [s] to finish the current phase."""
    cols = [i for i, a in enumerate(meta["branch_arms"]) if a == ARMS[arm]]
    return ep["branch_ticks"][:, cols].mean(1) * meta["tick_seconds"]


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    pad = w // 2
    c = np.cumsum(np.r_[0.0, x])
    lo = np.clip(np.arange(len(x)) - pad, 0, len(x))
    hi = np.clip(np.arange(len(x)) + pad + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def necessity(ep: dict, meta: dict, arm: int, window: int) -> np.ndarray:
    """(T,) extra time [s] the arm costs the current phase over the unmodified copies,
    smoothed over `window` ticks."""
    return smooth(np.maximum(cost(ep, meta, arm) - cost(ep, meta, AGAIN), 0.0), window)


def shares(ep: dict) -> tuple[np.ndarray, np.ndarray]:
    """(T,) the share of the executed action each warp side authors: the exact two player
    (Shapley) split of executed - skill between the input and the output warp, relative to
    the diagonal of the action range."""
    a_s, a_x = ep["skill_action"], ep["executed_action"]
    a_in, a_out = ep["input_warped_action"], ep["output_warped_action"]
    full = 2.0 * np.sqrt(a_s.shape[1])
    d_in = 0.5 * ((a_in - a_s) + (a_x - a_out))
    d_out = 0.5 * ((a_out - a_s) + (a_x - a_in))
    return np.linalg.norm(d_in, axis=1) / full, np.linalg.norm(d_out, axis=1) / full


def phase_table(meta: dict, episodes: list) -> dict:
    """Per phase, the mean over every tick spent in it: the time [s] the unmodified copies
    take to finish it, and the extra time each arm costs over them."""
    phase = np.concatenate([ep["phase"] for ep in episodes])
    again = np.concatenate([cost(ep, meta, AGAIN) for ep in episodes])
    extra = {ARMS[arm]: np.concatenate([cost(ep, meta, arm) for ep in episodes]) - again
             for arm in (OFF, IN_OFF, OUT_OFF, PULSE_OFF)}
    return {name: {"ticks": int(np.sum(phase == p)), ARMS[AGAIN]: float(again[phase == p].mean()),
                   **{arm: float(x[phase == p].mean()) for arm, x in extra.items()}}
            for p, name in enumerate(PHASES)}


def pick_frames(frames: np.ndarray, ticks: np.ndarray, T: int, k: int) -> list:
    """k frames at evenly spaced episode times -> [(tick, image)]."""
    wanted = np.linspace(0, T - 1, k)
    return [(int(ticks[j]), frames[j]) for j in (np.abs(ticks[None] - wanted[:, None])).argmin(1)]


def action_box(frames: np.ndarray, margin: int) -> tuple[slice, slice]:
    """The bounding box of the saturated (cube and jaw) pixels over the whole episode."""
    sat = frames.max(-1).astype(np.int16) - frames.min(-1).astype(np.int16)
    ys, xs = np.nonzero((sat > 70).any(0))
    H, W = frames.shape[1:3]
    return (slice(max(ys.min() - margin, 0), min(ys.max() + margin, H)),
            slice(max(xs.min() - margin, 0), min(xs.max() + margin, W)))


def shade(ax, phase: np.ndarray, placed: np.ndarray, t: np.ndarray, tau: float) -> None:
    edges = np.flatnonzero(np.r_[True, phase[1:] != phase[:-1], True])
    for k0, k1 in zip(edges[:-1], edges[1:]):
        ax.axvspan(t[k0] - tau, t[k1 - 1], color=PHASE_COLORS[PHASES[phase[k0]]], alpha=0.25,
                   lw=0)
    for k in np.flatnonzero(np.diff(placed) > 0) + 1:
        ax.axvline(t[k], color="black", lw=0.7, ls=":")


def figure(out_dir: str, meta: dict, ep: dict, n_frames: int, margin: int, window: int,
           out: str) -> None:
    """The three row figure of one episode: scene views, necessity, authored share."""
    fr = np.load(os.path.join(out_dir, f"frames_ep{ep['episode']:02d}.npz"))
    tau = meta["tick_seconds"]
    T = len(ep["phase"])
    t = np.arange(T) * tau

    fig = plt.figure(figsize=(9.5, 4.6))
    gs = fig.add_gridspec(3, n_frames, height_ratios=(1.35, 1.0, 1.0),
                          hspace=0.16, wspace=0.04, left=0.085, right=0.995, top=0.945,
                          bottom=0.20)
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[2, :], sharex=ax1)

    box = action_box(fr["frames"], margin)
    for k, (tick, img) in enumerate(pick_frames(fr["frames"], fr["ticks"], T, n_frames)):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img[box])
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_edgecolor(PHASE_COLORS[PHASES[ep["phase"][tick]]])
            side.set_linewidth(2.0)
        ax.set_title(f"{tick * tau:.1f} s", fontsize=9.5, pad=1.5)
        con = matplotlib.patches.ConnectionPatch(
            xyA=(0.5, -0.02), coordsA=ax.transAxes, xyB=(tick * tau, 1.0),
            coordsB=ax1.get_xaxis_transform(), color="0.55", lw=0.6, ls=(0, (2, 2)))
        fig.add_artist(con)

    shade(ax1, ep["phase"], ep["n_satisfied"], t, tau)
    ax1.plot(t, necessity(ep, meta, PULSE_OFF, window), color="0.45", lw=1.0,
             ls=(0, (2, 1.5)))
    ax1.plot(t, necessity(ep, meta, OFF, window), color="black", lw=1.5)
    ax1.plot(t, necessity(ep, meta, IN_OFF, window), color=IN_C, lw=1.1)
    ax1.plot(t, necessity(ep, meta, OUT_OFF, window), color=OUT_C, lw=1.1)
    ax1.set_ylabel("necessity [s]", fontsize=11)
    ax1.set_ylim(bottom=0.0)
    plt.setp(ax1.get_xticklabels(), visible=False)

    shade(ax2, ep["phase"], ep["n_satisfied"], t, tau)
    d_in, d_out = shares(ep)
    ax2.plot(t, d_in, color=IN_C, lw=1.1)
    ax2.plot(t, d_out, color=OUT_C, lw=1.1)
    ax2.set_ylabel("share of\nexecuted action", fontsize=11)
    ax2.set_ylim(bottom=0.0)
    ax2.set_xlabel("episode time [s]", fontsize=11)
    ax2.set_xlim(0.0, t[-1])

    handles = [
        matplotlib.lines.Line2D([], [], color="black", lw=1.5, label="full warp off"),
        matplotlib.lines.Line2D([], [], color="0.45", lw=1.0, ls=(0, (2, 1.5)),
                                label=f"full warp, {meta['pulse_ticks'] * tau:g} s gap"),
        matplotlib.lines.Line2D([], [], color=IN_C, lw=1.2,
                                label="$W^{\\mathrm{in}}$ off"),
        matplotlib.lines.Line2D([], [], color=OUT_C, lw=1.2,
                                label="$W^{\\mathrm{out}}$ off"),
        *[matplotlib.patches.Patch(color=PHASE_COLORS[p], alpha=0.45, label=p) for p in PHASES],
        matplotlib.lines.Line2D([], [], color="black", lw=0.7, ls=":", label="cube placed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=8.5, frameon=False,
               columnspacing=0.9, handlelength=1.3, handletextpad=0.5, bbox_to_anchor=(0.5, 0.005))

    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=250)
    plt.close(fig)


def summary(meta: dict, ep: dict, window: int) -> str:
    """One line per episode: how long it is and where its necessity sits, so a gallery can
    be compared without opening every figure."""
    full = necessity(ep, meta, OFF, window)
    out = necessity(ep, meta, OUT_OFF, window)
    gap = necessity(ep, meta, PULSE_OFF, window)
    release = ep["phase"] == RELEASE
    return (f"ep {ep['episode']:2d}  {len(ep['phase']) * meta['tick_seconds']:5.1f} s  "
            f"peak full {full.max():5.2f} s  peak W_out {out.max():5.2f} s  "
            f"W_out at releases {out[release].mean():5.2f} s  "
            f"gap at releases {gap[release].mean():5.2f} s  "
            f"quiet elsewhere {full[~release].mean():5.2f} s")


def report(args) -> None:
    """The per-phase table over every episode and the figure of every rendered episode."""
    plt.rcParams |= {"font.size": 11, "axes.linewidth": 0.7}
    meta, episodes = load(args.out_dir)
    tau = meta["tick_seconds"]
    table = phase_table(meta, episodes)
    with open(os.path.join(args.out_dir, "phases.json"), "w") as fh:
        json.dump(table, fh, indent=1)
    columns = [ARMS[a] for a in (AGAIN, OFF, IN_OFF, OUT_OFF, PULSE_OFF)]
    print("=" * 78)
    print(f"== WARP NECESSITY — {meta['task']}, {meta['n_objects']} objects, "
          f"{meta['episodes']} episodes, success {np.mean([ep['success'] for ep in episodes]):.2f}")
    print(f"==   {meta['replicas']} copies per arm, horizon {meta['horizon'] * tau:g} s, pulse "
          f"{meta['pulse_ticks'] * tau:g} s; mean over the ticks of each phase: the time [s] the "
          f"unmodified copies take, then the extra time each arm costs")
    print(f"==   {'phase':<17} " + " ".join(f"{c:>17}" for c in columns))
    for name, row in table.items():
        print(f"==   {name:<17} " + " ".join(f"{row[c]:17.2f}" for c in columns))
    print("=" * 78)

    rendered = [ep for ep in episodes
                if os.path.exists(os.path.join(args.out_dir, f"frames_ep{ep['episode']:02d}.npz"))]
    if args.figure_episode is not None:
        rendered = [ep for ep in rendered if ep["episode"] == args.figure_episode]
    for ep in rendered:
        out = os.path.join(args.out_dir, f"necessity_ep{ep['episode']:02d}")
        figure(args.out_dir, meta, ep, args.n_frames, args.margin, args.smooth_window, out)
        print(f"{summary(meta, ep, args.smooth_window)}   -> {os.path.basename(out)}.pdf/.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=(
        "When in an episode each side of the warp is needed: every tick of a deployed WAVE "
        "episode is branched into copies that run with the warper, its input side or its "
        "output side switched off, and the extra time each copy takes to finish the current "
        "phase is recorded. Prints the per-phase table and draws one figure per rendered "
        "episode (the paper's is episode 20)."))
    parser.add_argument("task", help="the WAVE task the head was trained on, e.g. gantry_towers_h6")
    parser.add_argument("checkpoint", help="the WAVE head, e.g. data/heads/gantry_towers_h6_seed42.pt")
    parser.add_argument("overrides", nargs="*", help="task config overrides")
    parser.add_argument("--out-dir", default="data/necessity")
    parser.add_argument("--report-only", action="store_true",
                        help="redraw the table and figures from the recording in --out-dir")
    parser.add_argument("--n-objects", type=int, default=4)
    parser.add_argument("--pool-seed", type=int, default=0, help="layout pool of the episodes")
    parser.add_argument("--episodes", type=int, default=32, help="independent episodes")
    parser.add_argument("--replicas", type=int, default=6,
                        help="branched copies per arm and tick")
    parser.add_argument("--horizon", type=int, default=120,
                        help="ticks a copy runs; a phase not finished by then counts the whole "
                             "horizon")
    parser.add_argument("--pulse-ticks", type=int, default=10,
                        help="ticks the 'warper off pulse' arm stays off before the warper "
                             "resumes")
    parser.add_argument("--render-episodes", type=int, default=8,
                        help="episodes built in one pass whose scene views are rendered")
    parser.add_argument("--figure-episode", type=int, default=None,
                        help="draw only this episode (default: every rendered one)")
    parser.add_argument("--n-frames", type=int, default=9, help="scene views above the curves")
    parser.add_argument("--margin", type=int, default=45,
                        help="pixels kept around the episode's cube bounding box")
    parser.add_argument("--at", type=float, default=1.0,
                        help="xy distance, in object heights, within which the hand is AT an "
                             "object: aligned to take it")
    parser.add_argument("--under", type=float, default=2.0,
                        help="xy distance, in object heights, within which an object is UNDER "
                             "the hand: the carry test's own reach")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="ticks of the centred window the drawn necessity is smoothed over")
    args = parser.parse_args()
    matplotlib.use("Agg")
    if not args.report_only:
        record(args)
    report(args)


if __name__ == "__main__":
    main()
