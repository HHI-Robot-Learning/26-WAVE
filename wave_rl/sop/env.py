import math

import mujoco
import numpy as np
import torch
import warp as wp

from wave_rl.config import PlantConfig, SkillRewardConfig
from wave_rl.sim.plant import Plant
from wave_rl.sim.sim import Sim

#: the goal feature: the object centre and its four corner markers, relative to the last
GOAL_FEAT_POINTS = ("obj", "goal1", "goal2", "goal3", "goal4", "obj_base")
#: bodies whose positions the skill observes, relative to the last
OBSERVATION_POINTS = ("goal1", "goal2", "goal3", "goal4", "obj_base")


class SkillEnv(Plant):
    """The single-object environment a skill is trained in: move `obj` so that its goal
    feature matches a goal configuration's. Episodes start from and aim at the rows the
    curriculum hands to :meth:`set_starts_goals`, in order."""

    def __init__(self, sim: Sim, plant: PlantConfig, reward: SkillRewardConfig) -> None:
        super().__init__(sim, plant)
        self.reward = reward
        B = self.num_scenes
        self.goal_feat_ids = torch.as_tensor(sim.body_ids(GOAL_FEAT_POINTS), dtype=torch.long,
                                             device=self.device)
        self.observation_ids = torch.as_tensor(sim.body_ids(OBSERVATION_POINTS),
                                               dtype=torch.long, device=self.device)
        bounds = np.array(sim.scene.limits, dtype=np.float64)
        for j in np.flatnonzero(sim.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            a = sim.model.jnt_qposadr[j]
            bounds[0, a + 3:a + 7] = -np.inf
            bounds[1, a + 3:a + 7] = np.inf
        self.bounds = torch.as_tensor(bounds, dtype=torch.float32, device=self.device)

        self.time_limit = 1.0
        self.needs_reset = torch.ones(B, dtype=torch.bool, device=self.device)
        self.scene_time = torch.zeros(B, device=self.device)
        self.goal_feat = torch.zeros((B, 3 * (len(GOAL_FEAT_POINTS) - 1)), device=self.device)
        self.observation, self.feat = self.observation_fct(self.goal_feat)
        self.starts_goals_counter = 0
        self.nan_resets = torch.zeros((), dtype=torch.long, device=self.device)
        self._prev_grasped = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._prev_reach = torch.zeros(B, device=self.device)
        self._prev_pregrasp = torch.zeros(B, device=self.device)
        print(f"-- skill env: observation dim {tuple(self.observation.shape)}, action dim "
              f"{self.action_dim}, tau step {plant.tau_step}")

    def set_starts_goals(self, starts_q: np.ndarray, goals_q: np.ndarray) -> None:
        """The episode list: start configurations and goal configurations, row-paired."""
        self.starts_goals_counter = 0
        self.starts_q = torch.as_tensor(starts_q, dtype=torch.float32, device=self.device)
        self.goals_feat = self.goal_features(torch.as_tensor(goals_q, dtype=torch.float32,
                                                             device=self.device))

    def goal_features(self, goals_q: torch.Tensor) -> torch.Tensor:
        """(n, F) the goal feature of each configuration, by batched forward kinematics on the
        simulator's own worlds; the simulator state is restored afterwards."""
        W, n = self.num_scenes, goals_q.shape[0]
        out = torch.empty((n, self.goal_feat.shape[1]), device=self.device)
        saved = self.sim.qpos_t.clone()
        write = torch.zeros_like(saved)
        with wp.ScopedStream(self.sim._wp_stream):
            for start in range(0, n, W):
                chunk = min(W, n - start)
                write[:chunk] = goals_q[start:start + chunk]
                self.sim.write(self.sim.data.qpos, write)
                self.sim.forward()
                out[start:start + chunk] = self._points(self.goal_feat_ids)[:chunk]
            self.sim.write(self.sim.data.qpos, saved)
            self.sim.forward()
        return out

    def auto_reset(self) -> torch.Tensor:
        """Reset the scenes that finished to the next start/goal pairs of the list."""
        mask = self.needs_reset
        n = int(mask.sum().item())
        if n == 0:
            return self.observation
        i = torch.arange(self.starts_goals_counter, self.starts_goals_counter + n,
                         device=self.device) % self.starts_q.shape[0]
        self.starts_goals_counter += n
        starts = self.starts_q[i]

        qpos, qvel, act = self.qpos.clone(), self.qvel.clone(), self.act.clone()
        qpos[mask] = starts
        qvel[mask] = 0.
        act[mask] = 0.
        self.goal_feat[mask] = self.goals_feat[i]
        self.ctrl_current[mask] = starts[:, self.ctrl_indices_t]
        self.sim.set_state(qpos, qvel, act, kinematics_only=True)
        self.sim.clear_warmstart(mask)

        obs, feat = self.observation_fct(self.goal_feat)
        self.observation = torch.where(mask.unsqueeze(1), obs, self.observation)
        self.feat = torch.where(mask.unsqueeze(1), feat, self.feat)
        self.needs_reset = torch.zeros_like(mask)
        self.scene_time[mask] = 0.
        if self.cfg.grip_binary:
            self.grip_closed = torch.where(mask, self.grip_closedness() > 0.25,
                                            self.grip_closed)
            self._prev_grasped = torch.where(mask, self._grasp_gate(feat) > 0.5,
                                             self._prev_grasped)
        if self.reward.reach_scale > 0.:
            self._prev_reach = torch.where(mask, self._reach_closeness(feat), self._prev_reach)
        if self.reward.pregrasp_scale > 0.:
            self._prev_pregrasp = torch.where(mask, self._pregrasp_closeness(feat),
                                              self._prev_pregrasp)
        return self.observation

    def reset(self) -> torch.Tensor:
        self.needs_reset[:] = True
        return self.auto_reset()

    def step(self, action: torch.Tensor):
        super().step(action)
        self.scene_time = self.scene_time + self.cfg.tau_step
        feat_prev = self.feat
        self.observation, self.feat = self.observation_fct(self.goal_feat)
        reward, terminated = self.reward_fct(self.feat, feat_prev)
        truncated = (self.scene_time >= self.time_limit) | self.out_of_bounds()

        finite = torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1)
        self.sim.heal(~finite)
        truncated = truncated | ~finite
        terminated = terminated & finite
        reward = torch.where(finite, reward, torch.zeros_like(reward))
        self.observation = torch.nan_to_num(self.observation, nan=0., posinf=0., neginf=0.)
        self.feat = torch.nan_to_num(self.feat, nan=0., posinf=0., neginf=0.)
        self.nan_resets += (~finite).sum()
        self.needs_reset = terminated | truncated
        return self.observation, reward, terminated, truncated

    def goal_distance(self, feat: torch.Tensor) -> torch.Tensor:
        """Position-first distance to the goal: sqrt(d_pos^2 + (w d_ori)^2)."""
        d_pos, d_ori = self._pos_ori_dist(feat)
        return torch.sqrt(d_pos ** 2 + (self.reward.ori_weight * d_ori) ** 2)

    def _pos_ori_dist(self, feat: torch.Tensor):
        """The goal distance split into position (centre L2) and orientation (corner markers
        relative to the centre, so they carry only rotation)."""
        n = feat.shape[0]
        obj, objg = feat[:, :3], self.goal_feat[:, :3]
        cor = feat[:, 3:].reshape(n, -1, 3) - obj[:, None, :]
        corg = self.goal_feat[:, 3:].reshape(n, -1, 3) - objg[:, None, :]
        d_pos = torch.linalg.norm(obj - objg, dim=1)
        k = self.reward.yaw_symmetry
        if k > 1:
            d_ori = _yaw_symmetric_ori_dist(cor, corg, k)
        else:
            d_ori = torch.linalg.norm((cor - corg).reshape(n, -1), dim=1)
        return d_pos, d_ori

    def reward_fct(self, feat: torch.Tensor, feat_prev: torch.Tensor):
        """1 at the goal, plus potential-based shaping of the position, the orientation and
        (near the goal) the combined distance, a grasp bonus, a drop penalty, and potentials
        pulling the hand to the object and closing it on it."""
        rw = self.reward
        dist = self.goal_distance(feat)
        is_goal = dist <= rw.eps_pos
        r = is_goal.float()

        d_pos, d_ori = self._pos_ori_dist(feat)
        d_pos_p, d_ori_p = self._pos_ori_dist(feat_prev)
        r = r + rw.pos_scale * (_ramp(d_pos, rw.pos_reach) - _ramp(d_pos_p, rw.pos_reach))
        ori_now, ori_prev = _ramp(d_ori, rw.ori_reach), _ramp(d_ori_p, rw.ori_reach)
        if rw.ori_gate is not None:
            ori_now = ori_now * _ramp(d_pos, rw.ori_gate)
            ori_prev = ori_prev * _ramp(d_pos_p, rw.ori_gate)
        r = r + rw.ori_scale * (ori_now - ori_prev)
        dist_prev = self.goal_distance(feat_prev)
        r = r + rw.combined_scale * (_ramp(dist, rw.combined_reach)
                                     - _ramp(dist_prev, rw.combined_reach))

        if rw.reach_scale > 0.:
            close = self._reach_closeness(feat)
            r = r + rw.reach_scale * (close - self._prev_reach)
            self._prev_reach = close
        if rw.pregrasp_scale > 0.:
            close = self._pregrasp_closeness(feat)
            r = r + rw.pregrasp_scale * (close - self._prev_pregrasp)
            self._prev_pregrasp = close
        if rw.grasp > 0. or rw.release > 0.:
            gate = self._grasp_gate(feat)
            grasped = gate > 0.5
            dropped = self._prev_grasped & ~grasped & ~is_goal
            self._prev_grasped = grasped
            r = r + rw.grasp * gate - rw.release * dropped.float()
        return r, is_goal

    def _reach_closeness(self, feat):
        """Potential in [0, 1] for bringing the hand to the object."""
        reach = self.reward.reach_len or 10. * self.reward.eps_pos
        return _ramp(torch.linalg.norm(feat[:, :3] - self.hand_position(), dim=1), reach)

    def _pregrasp_closeness(self, feat):
        """Potential in [0, 1] for closing the jaw while the hand is at the object."""
        reach = self.reward.pregrasp_len or 4. * self.reward.eps_pos
        near = _ramp(torch.linalg.norm(feat[:, :3] - self.hand_position(), dim=1), reach)
        return near * self.grip_closedness()

    def _grasp_gate(self, feat):
        """Soft grasp detector in [0, 1]: the object lifted off its rest height and centred
        under the hand in xy."""
        rw = self.reward
        lift = ((feat[:, 2] - rw.grasp_rest_z) / rw.grasp_lift_scale).clamp(0., 1.)
        d_xy2 = ((feat[:, :2] - self.hand_xy()) ** 2).sum(dim=1)
        return lift * torch.exp(-d_xy2 / (2. * rw.grasp_reach_sigma ** 2))

    def observation_fct(self, goal_feat: torch.Tensor):
        """[ joint positions without the object quaternion | velocities | servo error |
        marker positions | goal feature - feature ], positions relative to `obj_base`."""
        qpos, qvel = self.qpos, self.qvel
        feat = self._points(self.goal_feat_ids)
        obs = torch.cat([qpos[:, :-4],
                         self.reward.vel_scale * qvel,
                         self.reward.referr_scale * (self.ctrl_current
                                                     - qpos[:, self.ctrl_indices_t]),
                         self._points(self.observation_ids),
                         goal_feat - feat], dim=1)
        return obs, feat

    def _points(self, ids):
        """(B, 3 (k - 1)) positions of bodies `ids[:-1]` relative to body `ids[-1]`."""
        pts = self.sim.xpos_t[:, ids]
        return (pts[:, :-1] - pts[:, -1:]).reshape(self.num_scenes, -1)

    def out_of_bounds(self) -> torch.Tensor:
        m = self.reward.bounds_margin
        return ((self.qpos < self.bounds[0] - m) | (self.qpos > self.bounds[1] + m)).any(dim=1)


def _ramp(d: torch.Tensor, reach: float) -> torch.Tensor:
    """1 at distance 0, falling linearly to 0 at `reach`."""
    return (1. - d / reach).clamp(min=0.)


def _yaw_symmetric_ori_dist(cor, corg, k):
    """Orientation distance modulo the object's k-fold yaw symmetry: the closest of the k
    copies of the goal markers spun about world z."""
    n = cor.shape[0]
    ang = torch.arange(k, device=cor.device, dtype=cor.dtype) * (2. * math.pi / k)
    c, s = torch.cos(ang), torch.sin(ang)
    x, y, z = corg[..., 0], corg[..., 1], corg[..., 2]
    rx = x[:, None, :] * c[None, :, None] - y[:, None, :] * s[None, :, None]
    ry = x[:, None, :] * s[None, :, None] + y[:, None, :] * c[None, :, None]
    spun = torch.stack([rx, ry, z[:, None, :].expand_as(rx)], dim=-1)
    d = torch.linalg.norm((cor[:, None] - spun).reshape(n, k, -1), dim=2)
    return d.amin(dim=1)
