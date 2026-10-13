import numpy as np
import torch

from wave_rl.config import GoalConfig, PlantConfig
from wave_rl.env.goals import make_task_goal
from wave_rl.env.observation import BatchedObservation
from wave_rl.sim.plant import Plant
from wave_rl.sim.scene import Scene, arena_bound
from wave_rl.sim.sim import Sim
from wave_rl.skills.kinematics import build_joint_chain

#: Column layout of one finished-episode row in the device ring buffer.
(_SUCCESS, _SATISFIED, _TICKS, _COUNT, _TIMEOUT, _LOST, _SIMFAIL, _SEQ) = range(8)
_N_COLS = 8


class CountSchedule:
    """How many objects each scene gets per episode — the one place the count policy lives,
    so the env is just a stepping engine that asks it for counts.

    Modes:
      - ``fixed`` : every scene at ``n_max``.
      - ``mixed`` : uniform in ``[1, n_max]``, the training schedule of every task.
    """

    def __init__(self, mode: str, n_max: int, gen: torch.Generator, device) -> None:
        if mode not in ("fixed", "mixed"):
            raise ValueError(f"unknown count_mode {mode!r}; supported: 'fixed', 'mixed'")
        self.mode = mode
        self.n_max = n_max
        self._gen = gen
        self._dev = device
        self._override = None

    def set_override(self, count: int) -> None:
        """Pin every scene's active object count, which is how the eval sweep rolls one
        count at a time on one scene."""
        self._override = count

    def sample_counts(self, B: int) -> torch.Tensor:
        """Per-scene active object count ``(B,)`` long, on device."""
        if self._override is not None:
            return torch.full((B,), self._override, device=self._dev, dtype=torch.long)
        if self.mode == "fixed":
            return torch.full((B,), self.n_max, device=self._dev, dtype=torch.long)
        return torch.randint(1, self.n_max + 1, (B,), generator=self._gen, device=self._dev)


class EpisodeStatistics:
    """Per-object-count outcome statistics of the TRAINING rollout itself.

    Every quantity an interleaved evaluation would measure is already visible in the training
    loop: each scene runs episodes back to back, and ``active_count`` says how many objects
    that episode had. Bucketing finished episodes by that count gives a full count curve
    without ever pausing training.

    **No host sync in the step path.** Per-episode accumulators live on device and finished
    episodes are scattered into a device ring buffer using a cumsum-derived slot, so
    :meth:`observe` costs nothing per tick. The single sync happens in :meth:`metrics`, which
    the training loop calls once per logging block.

    These are the statistics of the exploring policy. The eval sweep runs the greedy policy
    and reads systematically higher, which is why these live under ``ep/`` and the sweep's
    under ``sweep/``.
    """

    def __init__(self, num_scenes: int, max_objects: int, device, window: int) -> None:
        #: episodes retained per object count in the rolling window
        self.window = window
        B = num_scenes
        # The ring must outlive one step's worth of simultaneous terminations (B of them)
        # and hold enough history to fill every count's window.
        self.capacity = max(4 * B, self.window * max_objects * 4, 4096)
        self._dev = device
        # slot `capacity` is a trash row: scenes that did NOT finish this tick all write
        # there, so the live slots are touched by exactly one scene each and the scatter
        # needs no mask (and therefore no host sync) to stay deterministic.
        self._rows = torch.zeros((self.capacity + 1, _N_COLS), device=device,
                                 dtype=torch.float32)
        self._written = torch.zeros((), device=device, dtype=torch.long)
        self._ticks = torch.zeros(B, device=device, dtype=torch.long)

    @torch.no_grad()
    def observe(self, satisfied: torch.Tensor, active: torch.Tensor,
                active_count: torch.Tensor, terminated: torch.Tensor,
                truncated: torch.Tensor, lost: torch.Tensor,
                sim_failure: torch.Tensor) -> None:
        """Fold one env step into the accumulators and retire finished episodes."""
        cur_n = (satisfied & active).sum(dim=1)
        self._ticks = self._ticks + 1

        done = terminated | truncated
        pos = torch.cumsum(done.long(), dim=0) - 1
        slot = torch.where(done, (self._written + pos) % self.capacity,
                           torch.full_like(pos, self.capacity))
        # A truncation has exactly three causes and they are NOT interchangeable: the clock
        # running out is a policy verdict, a lost object or a NaN world is the plant ending
        # the episode early. Score them apart.
        row = torch.stack([
            terminated.float(), cur_n.float(), self._ticks.float(), active_count.float(),
            (truncated & ~lost & ~sim_failure).float(), lost.float(), sim_failure.float(),
            (self._written + pos).float(),
        ], dim=1)
        self._rows[slot] = row
        self._written = self._written + done.long().sum()

        self._ticks = self._ticks * ~done

    @torch.no_grad()
    def metrics(self) -> dict:
        """Per-count aggregates over the rolling window. ONE host sync. Every count present
        in the window is reported, so a mixed-count run needs no configuration to produce its
        whole curve."""
        written = self._written.item()
        if not written:
            return {}
        rows = self._rows[:self.capacity].to("cpu").numpy()
        if written < self.capacity:
            rows = rows[:written]
        # Newest first, so a per-count head slice is the most recent `window` episodes.
        rows = rows[np.argsort(-rows[:, _SEQ], kind="stable")]

        out: dict[str, float] = {}
        for c in sorted({int(c) for c in rows[:, _COUNT].tolist()}):
            sel = rows[rows[:, _COUNT] == c][:self.window]
            out[f"ep/full_success_rate@{c}obj"] = float(sel[:, _SUCCESS].mean())
            out[f"ep/mean_satisfied_at_episode_end@{c}obj"] = float(sel[:, _SATISFIED].mean())
            out[f"ep/mean_ticks_per_episode@{c}obj"] = float(sel[:, _TICKS].mean())
            out[f"ep/episode_end_timeout_fraction@{c}obj"] = float(sel[:, _TIMEOUT].mean())
            out[f"ep/episode_end_lost_object_fraction@{c}obj"] = float(sel[:, _LOST].mean())
            out[f"ep/episode_end_sim_failure_fraction@{c}obj"] = float(sel[:, _SIMFAIL].mean())
            out[f"ep/episodes_in_window@{c}obj"] = float(len(sel))
        out["ep/episodes_finished_total"] = float(written)
        return out


def resolve_layout(plant: Plant, obj_names: list[str], ref_name: str) -> dict:
    """Where every object, the reference frame and the end effector live in the compiled
    model. Every world is the same scene, so body ids are shared and a scene's positions live
    in the world of its own index."""
    model = plant.sim.model
    qpos_start, vel_start = [], []
    for name in obj_names:
        body = model.body(name).id
        joint = int(model.body_jntadr[body])
        assert joint >= 0 and model.body_jntnum[body] == 1, (
            f"body {name!r} must carry exactly one free joint")
        qpos_start.append(int(model.jnt_qposadr[joint]))
        vel_start.append(int(model.jnt_dofadr[joint]))

    nq, nv = plant.qpos.shape[1], plant.qvel.shape[1]
    object_q = {i for qs in qpos_start for i in range(qs, qs + 7)}
    object_v = {i for vs in vel_start for i in range(vs, vs + 6)}
    ee_name = plant.cfg.grasp_ee_point
    return {
        "obj_qpos_start": qpos_start,
        "obj_vel_start": vel_start,
        "body_ids": [model.body(n).id for n in obj_names],
        "ref_id": model.body(ref_name).id,
        "ee_id": None if ee_name is None else model.body(ee_name).id,
        "robot_qpos": np.array(sorted(set(range(nq)) - object_q), dtype=np.int32),
        "robot_qvel": np.array(sorted(set(range(nv)) - object_v), dtype=np.int32),
    }


class MultiObjectEnv:
    """GPU-batched multi-object env: ``B`` scenes stepped together on one
    :class:`~wave_rl.sim.plant.Plant`.

    The **entire step runs on the sim's device with no host sync**: physics, the
    :class:`~wave_rl.env.goals.StructureGoal` predicate, completion freezing and auto-reset
    are all torch tensor ops over ``(B, N)`` device tensors. Task state lives in device
    tensors — there is no host mirror to sync. Auto-reset is BRANCH-FREE: a fresh episode is
    sampled for every scene each step (cheap on device) and applied only where ``dones`` via
    ``torch.where``, so nothing ever has to read which scenes finished back to the host.

    It emits a single :class:`~wave_rl.env.observation.BatchedObservation` — objects padded
    to ``N`` and masked — which is the one seam every algorithm consumes.

    It owns **nothing about what "satisfied" means**: the injected `StructureGoal` decides that.
    """

    def __init__(self, plant: Plant, scene: Scene, task_goal, configs_q: np.ndarray,
                 num_scenes: int, goal_cfg: GoalConfig, count_mode: str,
                 episode_time_limit: float, time_per_object: float | None,
                 remove_on_completion: bool, seed: int) -> None:
        self.plant = plant
        self.scene = scene
        self.task_goal = task_goal
        self.num_scenes = num_scenes
        self.max_objects = len(scene.obj_names)
        self.action_dim = plant.action_dim
        self.episode_time_limit = episode_time_limit
        # if set, the per-episode time budget scales with the scene's active object count
        # (limit = time_per_object * active_count); else the flat episode_time_limit applies.
        self.time_per_object = time_per_object

        self.success_mode = goal_cfg.success_mode
        self.success_dwell = goal_cfg.success_dwell
        # Freeze completed objects only in the latched regime; in 'simultaneous' mode placed
        # objects must stay dynamic (and knockable), which is that mode's whole point.
        self.freeze_completed = remove_on_completion and self.success_mode != "simultaneous"

        dev = plant.device
        self.device = dev
        self._dtype = plant.qpos.dtype
        self.nq = plant.qpos.shape[1]
        self.nv = plant.qvel.shape[1]

        layout = resolve_layout(plant, list(scene.obj_names), scene.ref_name)
        self._body_ids = torch.tensor(layout["body_ids"], dtype=torch.long, device=dev)
        self._ref_id = layout["ref_id"]
        self._ee_id = layout["ee_id"]
        self._qpos_cols = torch.tensor(
            [[qs + d for d in range(7)] for qs in layout["obj_qpos_start"]],
            dtype=torch.long, device=dev)                                        # (N, 7)
        self._vel_cols = torch.tensor(
            [[vs + d for d in range(6)] for vs in layout["obj_vel_start"]],
            dtype=torch.long, device=dev)                                        # (N, 6)
        self._robot_q_idx = torch.as_tensor(layout["robot_qpos"], dtype=torch.long,
                                            device=dev)
        self._robot_v_idx = torch.as_tensor(layout["robot_qvel"], dtype=torch.long,
                                            device=dev)
        self._ctrl_idx = plant.ctrl_indices_t
        self._arange_n = torch.arange(self.max_objects, device=dev)
        self.tau_step = plant.cfg.tau_step
        self._prev_ee = None

        #: widths a frozen skill's projection is sized against
        self.n_robot_qpos = self._robot_q_idx.shape[0]
        self.n_robot_qvel = self._robot_v_idx.shape[0]
        self.n_ctrl = self._ctrl_idx.shape[0]

        # An active object past either bound can never be recovered, so the episode ends as
        # UNACHIEVABLE: done, not success, and NOT value-bootstrapped.
        self.lost_z = -0.3
        self.arena_xy = arena_bound(plant.sim.model)
        print(f"-- UNACHIEVABLE ABORT: lost_z={self.lost_z} arena_xy="
              f"{'OFF' if self.arena_xy is None else f'{self.arena_xy:.3f} m'} "
              f"(outer face of the outermost static containment)")

        self._configs_t = torch.as_tensor(configs_q, dtype=self._dtype, device=dev)
        self._n_rows = self._configs_t.shape[0]
        self._park_poses = self._make_park_poses()
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(seed)
        self.schedule = CountSchedule(count_mode, self.max_objects, self._gen, dev)

        B, N = self.num_scenes, self.max_objects
        self.active_count = torch.full((B,), N, device=dev, dtype=torch.long)
        self.nominal_goals_feat = torch.zeros((B, N, 3), device=dev, dtype=self._dtype)
        self.goals_feat = torch.zeros((B, N, 3), device=dev, dtype=self._dtype)
        self.goals_qpos = torch.zeros((B, N, 7), device=dev, dtype=self._dtype)
        self.completed = torch.zeros((B, N), device=dev, dtype=torch.bool)
        self.last_status = torch.zeros((B, N), device=dev, dtype=torch.bool)
        self._sat_streak = torch.zeros(B, device=dev, dtype=torch.long)
        self.episode_time = torch.zeros(B, device=dev, dtype=self._dtype)
        self._last_q_live = torch.zeros((B, self.nq), device=dev, dtype=self._dtype)
        self._nan_world_mask = torch.zeros(B, device=dev, dtype=torch.bool)
        self._nan_worlds_seen = 0

        # Training-rollout observers. Both are device-side and cost no host sync per tick,
        # and both stay None until a caller arms them.
        self.episode_stats = None
        self.video_capture = None

    def _make_park_poses(self) -> torch.Tensor:
        """Parking pose per object slot: stacked high above the arena, identity quat."""
        poses = torch.zeros((self.max_objects, 7), device=self.device, dtype=self._dtype)
        for k in range(self.max_objects):
            poses[k, 2] = 3.0 + 0.6 * k
            poses[k, 3] = 1.0
        return poses

    def _active_mask(self) -> torch.Tensor:
        """(B, N) bool — object j active in scene s iff j < active_count[s]."""
        return self._arange_n[None, :] < self.active_count[:, None]

    def _sample_episodes(self) -> tuple:
        """Draw a fresh episode for **every** scene on device (the caller applies it only
        where a scene is done). Returns ``(k, new_active, start_q, goal_feat, goal_qpos)``."""
        B = self.num_scenes
        k = self.schedule.sample_counts(B)
        new_active = self._arange_n[None, :] < k[:, None]

        start_idx = torch.randint(0, self._n_rows, (B,), generator=self._gen,
                                  device=self.device)
        start_q = self._configs_t[start_idx].clone()
        goal_idx = torch.randint(0, self._n_rows, (B,), generator=self._gen,
                                 device=self.device)
        # Draw a resting goal pose from the pool, then let the TASK GOAL place the actual
        # targets. The env stays goal-agnostic — see StructureGoal.place_goals.
        goal_obj = self.task_goal.place_goals(
            self._configs_t[goal_idx][:, self._qpos_cols], self._arange_n, new_active)

        am = new_active.unsqueeze(-1)
        start_q[:, self._qpos_cols] = torch.where(
            am, start_q[:, self._qpos_cols], self._park_poses[None])
        # Start the hand over object 0 on a Cartesian carriage, whose leading ctrl channels are
        # the hand's world x and y: immediate contact bootstraps the push reward, where a
        # neutral pose ~1.5 m away never contacts, so no reward ever arrives and the actor
        # saturates. An arm's leading ctrl channels are joint angles, and its pool row already
        # holds the sampled start pose, which must stay as drawn.
        if self._ee_id is None:
            start_q[:, self._ctrl_idx[:2]] = start_q[:, self._qpos_cols[0]][:, :2]

        goal_qpos = torch.where(am, goal_obj, torch.zeros_like(goal_obj))
        return k, new_active, start_q, goal_qpos[..., :3].clone(), goal_qpos

    def reset(self) -> BatchedObservation:
        """Sample a fresh episode for every scene; return the batched observation."""
        B = self.num_scenes
        k, new_active, start_q, goal_feat, goal_qpos = self._sample_episodes()
        self.active_count = k
        self.nominal_goals_feat = goal_feat
        self.goals_feat = goal_feat.clone()
        self.goals_qpos = goal_qpos
        self.completed = torch.zeros((B, self.max_objects), device=self.device,
                                     dtype=torch.bool)
        self.last_status = torch.zeros((B, self.max_objects), device=self.device,
                                       dtype=torch.bool)
        self._sat_streak = torch.zeros(B, device=self.device, dtype=torch.long)
        self.episode_time = torch.zeros(B, device=self.device, dtype=self._dtype)

        v = torch.zeros((B, self.nv), device=self.device, dtype=self._dtype)
        self.plant.sim.set_state(start_q, v)
        self.plant.ctrl_current[:] = start_q[:, self._ctrl_idx]

        # xpos is stale after set_state on the warp backend (no forward); use the start
        # config's own translations for the fresh features — identical to how goal features
        # are defined, and correct on both backends.
        feats = start_q[:, self._qpos_cols][..., :3]
        self._last_q_live = start_q
        self._prev_ee = None
        return self._build_obs(feats, start_q, v, self.plant.ctrl_current, new_active,
                               self.last_status)

    def step(self, actions) -> tuple:
        """Advance every scene with a ``(B, action_dim)`` action batch.

        Returns ``(obs, obs_true, terminated, truncated, info)``. ``obs`` is the
        POST-auto-reset state to act from next; ``obs_true`` is the pre-reset true next state,
        which is what a buffer must bootstrap from at a timeout — storing the post-reset
        observation (a fresh, unrelated episode) and treating the time limit as terminal caps
        the value horizon at the episode length. ``terminated`` and ``truncated`` stay
        distinct so the caller bootstraps timeouts but never real terminals.

        The env computes no reward: what a learner optimises is its own `SceneReward`
        (`wave_rl.reward`), stepped on `obs_true` and `obs`.
        """
        assert actions.shape == (self.num_scenes, self.action_dim), (
            f"actions must be ({self.num_scenes}, {self.action_dim}), got {actions.shape}")
        self.plant.step(actions)
        self.episode_time = self.episode_time + self.tau_step

        feats, speeds, q_live, v_live, ctrl = self._read()
        feats, speeds, q_live, v_live, ctrl = self._quarantine_nan_worlds(
            feats, speeds, q_live, v_live, ctrl)
        self._last_q_live = q_live

        active = self._active_mask()
        self.goals_feat = self.task_goal.compile_goals(
            self.nominal_goals_feat, feats, self.last_status, active)
        status = self.task_goal.satisfied(feats, self.goals_feat, speeds, active)
        self.last_status = status
        self.completed = self.completed | status

        if self.success_mode == "simultaneous":
            all_sat = (status | ~active).all(dim=1)
            self._sat_streak = torch.where(all_sat, self._sat_streak + 1,
                                           torch.zeros_like(self._sat_streak))
            terminated = self._sat_streak >= self.success_dwell
        else:   # latch_freeze: solved once every active object has EVER been reached
            terminated = (self.completed | ~active).all(dim=1)

        limit = (self.time_per_object * self.active_count.to(self._dtype)
                 if self.time_per_object is not None else self.episode_time_limit)
        lost_z = active & (feats[..., 2] < self.lost_z)
        lost_xy = (active & (feats[..., :2].abs().amax(dim=-1) > self.arena_xy)
                   if self.arena_xy is not None else torch.zeros_like(lost_z))
        unachievable = (lost_z | lost_xy).any(dim=1)
        truncated = (~terminated) & ((self.episode_time >= limit) | unachievable
                                     | self._nan_world_mask)
        terminated = terminated & ~self._nan_world_mask
        dones = terminated | truncated

        ac_info = self.active_count.clone()
        # Retire finished episodes into the rolling per-count window BEFORE the auto-reset
        # below overwrites active_count with the next episode's draw.
        if self.episode_stats is not None:
            self.episode_stats.observe(status, active, ac_info, terminated, truncated,
                                       unachievable & ~terminated, self._nan_world_mask)
        obs_true = self._build_obs(feats, q_live, v_live, ctrl, active, status)

        q, v = q_live.clone(), v_live.clone()
        q_obj = torch.where(active.unsqueeze(-1), q[:, self._qpos_cols],
                            self._park_poses[None])
        v_obj = torch.where(active.unsqueeze(-1), v[:, self._vel_cols],
                            torch.zeros_like(v[:, self._vel_cols]))
        if self.freeze_completed:
            cm = self.completed.unsqueeze(-1)
            q_obj = torch.where(cm, self.goals_qpos, q_obj)
            v_obj = torch.where(cm, torch.zeros_like(v_obj), v_obj)
        q[:, self._qpos_cols] = q_obj
        v[:, self._vel_cols] = v_obj

        # auto-reset: sample fresh for ALL scenes, apply only where done (branch-free)
        k_new, new_active, start_q, goal_feat_new, goal_qpos_new = self._sample_episodes()
        d1, d2 = dones[:, None], dones[:, None, None]
        q = torch.where(d1, start_q, q)
        v = torch.where(d1, torch.zeros_like(v), v)
        self.active_count = torch.where(dones, k_new, self.active_count)
        self.nominal_goals_feat = torch.where(d2, goal_feat_new, self.nominal_goals_feat)
        self.goals_feat = torch.where(d2, goal_feat_new, self.goals_feat)
        self.goals_qpos = torch.where(d2, goal_qpos_new, self.goals_qpos)
        self.completed = torch.where(d1, torch.zeros_like(self.completed), self.completed)
        self.last_status = torch.where(d1, torch.zeros_like(status), status)
        self._sat_streak = torch.where(dones, torch.zeros_like(self._sat_streak),
                                       self._sat_streak)
        self.episode_time = torch.where(dones, torch.zeros_like(self.episode_time),
                                        self.episode_time)

        # The jaw latch and the previous action carry across the episode boundary: the
        # first command of a new episode is read against the last one of the old.
        self.plant.sim.set_state(q, v)
        self.plant.ctrl_current = torch.where(d1, q[:, self._ctrl_idx], ctrl)

        active_obs = self._active_mask()
        feats_obs = torch.where(d2, start_q[:, self._qpos_cols][..., :3], feats)
        obs = self._build_obs(feats_obs, q, v, self.plant.ctrl_current, active_obs,
                              self.last_status, reset_mask=dones)

        # The capture runs AFTER the auto-reset: `q_live` is still the finished episode's
        # terminal state (the frame we want), while goals/active already describe the next
        # episode — which is what the capture needs to label the frames it starts storing.
        if self.video_capture is not None:
            self.video_capture.observe(q_live, self.goals_qpos, active_obs, dones)

        info = {"unachievable": unachievable & ~terminated,
                "sim_failure": self._nan_world_mask, "active_count": ac_info}
        return obs, obs_true, terminated, truncated, info

    def observe(self) -> BatchedObservation:
        feats, _speeds, q, v, ctrl = self._read()
        return self._build_obs(feats, q, v, ctrl, self._active_mask(), self.last_status,
                               store=False)

    def _read(self) -> tuple:
        """One batched device read (no host transfer): object features ``(B,N,3)``, linear
        speeds ``(B,N)``, full per-scene ``qpos``/``qvel`` and the control reference."""
        g = self.plant
        xpos = g.sim.xpos_t                                       # (B, nbody, 3)
        obj = xpos[:, self._body_ids]                             # (B, N, 3)
        ref = xpos[:, self._ref_id][:, None, :]                   # (B, 1, 3)
        speeds = torch.linalg.norm(g.qvel[:, self._vel_cols][..., :3], dim=-1)
        return obj - ref, speeds, g.qpos, g.qvel, g.ctrl_current

    def _quarantine_nan_worlds(self, feats, speeds, q_live, v_live, ctrl) -> tuple:
        """A world the simulator drove non-finite is unrecoverable: its state is restored to
        the previous tick's, its solver warm-start is zeroed, and it is marked so the episode
        ends as a sim failure rather than poisoning every reduction it takes part in."""
        bad = ~(torch.isfinite(q_live).all(dim=1) & torch.isfinite(v_live).all(dim=1))
        self._nan_world_mask = bad
        if not bool(bad.any()):
            return feats, speeds, q_live, v_live, ctrl
        n_bad = int(bad.sum())
        self._nan_worlds_seen += n_bad
        q_live = torch.nan_to_num(torch.where(bad[:, None], self._last_q_live, q_live),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        v_live = torch.where(bad[:, None], torch.zeros_like(v_live), v_live)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        speeds = torch.nan_to_num(speeds, nan=0.0, posinf=0.0, neginf=0.0)
        ctrl = torch.nan_to_num(ctrl, nan=0.0, posinf=0.0, neginf=0.0)
        self.plant.sim.heal(bad)
        print(f"!! NaN-world quarantine: {n_bad} of {self.num_scenes} worlds went "
              f"non-finite in the simulator and were reset "
              f"({self._nan_worlds_seen} total this run)")
        return feats, speeds, q_live, v_live, ctrl

    def attach_kinematics(self, skill) -> None:
        """Transcribe the frozen skill's declared joint chain out of THIS env's compiled
        model and hand it back.

        The env is the owner because the chain is a fact about the compiled plant — link
        offsets, joint axes, the qpos columns the observation publishes — none of which the
        skill can see from its snapshot. An embodiment that declares no `chain_joints` (a
        Cartesian carriage) is left alone.
        """
        if not skill.chain_joints:
            return
        skill.chain = build_joint_chain(
            self.plant.sim.model, skill.chain_joints, skill.ee_frame,
            self._robot_q_idx.cpu().numpy(), self.plant.action_scale, self.device)
        print(skill.chain.banner())

    def _build_obs(self, feats, q, v, ctrl, active, satisfied, reset_mask=None,
                   store: bool = True) -> BatchedObservation:
        """Assemble the device :class:`BatchedObservation` from in-hand tensors — no re-read,
        so it costs nothing and never forces a sync.

        The ONE exception is the end-effector block on an arm, which cannot come from qpos
        (that is the whole point) and so costs a single device gather out of ``xpos_t``. It is
        read HERE rather than in ``_read`` because ``_read`` runs BEFORE the auto-reset writes
        ``set_state``, and a scene that reset this tick would otherwise publish the previous
        episode's hand position. ``reset_mask`` zeroes those scenes' velocity so a fresh
        episode never differences across the discontinuity; ``store=False`` lets a read-only
        probe sample without perturbing the difference sequence."""
        ee_pos = ee_vel = None
        if self._ee_id is not None:
            xp = self.plant.sim.xpos_t
            ee_pos = xp[:, self._ee_id] - xp[:, self._ref_id]
            prev = self._prev_ee
            ee_vel = (torch.zeros_like(ee_pos) if prev is None
                      else (ee_pos - prev) / self.tau_step)
            if reset_mask is not None:
                ee_vel = torch.where(reset_mask[:, None], torch.zeros_like(ee_vel), ee_vel)
            if store:
                self._prev_ee = ee_pos
        return BatchedObservation(
            robot_qpos=q[:, self._robot_q_idx],
            robot_qvel=v[:, self._robot_v_idx],
            obj_qpos=q[:, self._qpos_cols],
            obj_qvel=v[:, self._vel_cols],
            obj_feat=feats,
            goal_feat=self.goals_feat,
            ctrl_ref=ctrl,
            ctrl_pos=q[:, self._ctrl_idx],
            active_mask=active,
            satisfied_mask=satisfied,
            ee_pos=ee_pos,
            ee_vel=ee_vel,
        )


def build_env(scene: Scene, plant_cfg: PlantConfig, goal_cfg: GoalConfig,
              configs_q: np.ndarray, num_scenes: int, count_mode: str,
              episode_time_limit: float, time_per_object: float | None,
              remove_on_completion: bool, seed: int = 0) -> MultiObjectEnv:
    """Wire a :class:`MultiObjectEnv` on ``scene``: one N-object scene replicated into
    ``num_scenes`` parallel worlds, the task goal built from the scene's own object geometry,
    and the ``configs_q`` rest-config pool the episode starts and goals are drawn from."""
    print(f"== CONTACTS: friction={plant_cfg.friction!r} solref={plant_cfg.solref!r} "
          f"solimp={plant_cfg.solimp!r} impratio={plant_cfg.impratio} "
          f"tau_sim={plant_cfg.tau_sim}")
    plant = Plant(Sim(scene, plant_cfg, num_scenes), plant_cfg)

    nq_scene = plant.qpos.shape[1]
    if configs_q.shape[1] != nq_scene:
        raise ValueError(
            f"config pool has q width {configs_q.shape[1]}, but the "
            f"{len(scene.obj_names)}-object scene has per-scene qpos dimension {nq_scene}.")

    # A lattice's slots are anchored at a drawn rest pose and extend from it, so they need a
    # bound on where an object can legally come to rest. Every pool row IS such a placement,
    # so the pool answers it with no geometry reasoning of its own.
    arm_nq = nq_scene - 7 * len(scene.obj_names)
    xy = configs_q[:, arm_nq:].reshape(len(configs_q), len(scene.obj_names), 7)[:, :, :2]
    task_goal = make_task_goal(goal_cfg, scene.object, float(np.abs(xy).max()))

    return MultiObjectEnv(
        plant=plant, scene=scene, task_goal=task_goal, configs_q=configs_q,
        num_scenes=num_scenes, goal_cfg=goal_cfg, count_mode=count_mode,
        episode_time_limit=episode_time_limit, time_per_object=time_per_object,
        remove_on_completion=remove_on_completion, seed=seed)
