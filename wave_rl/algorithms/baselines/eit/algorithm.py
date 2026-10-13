import math

import numpy as np
import torch

from wave_rl.algorithms.base import Algorithm
from wave_rl.env.observation import BatchedObservation
from wave_rl.td7.buffer import LAP

from .agent import EITAgent
from .codec import ObsCodec, RowLayout
from .config import EITConfig
from .distill import distill_from_teacher
from .her import HERRelabeler


class OUNoise:
    """Ornstein-Uhlenbeck action noise, per scene and reset at episode end: temporally
    correlated, so exploratory actions are sustained and directed (the gantry traverses and
    pushes) instead of white noise that self-cancels."""

    def __init__(self, num_scenes: int, action_dim: int, device, std: float, theta: float) -> None:
        self.theta = theta
        self.ou = torch.zeros(num_scenes, action_dim, device=device)
        #: so that the discrete process has stationary std == `std`
        self.sigma = std * math.sqrt(2.0 * theta - theta * theta)

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        self.ou = (1.0 - self.theta) * self.ou + self.sigma * torch.randn_like(self.ou)
        return (actions + self.ou).clamp(-1.0, 1.0)

    def reset(self, dones: torch.Tensor) -> None:
        self.ou = torch.where(dones.unsqueeze(1), torch.zeros_like(self.ou), self.ou)


class EIT(Algorithm):
    """Entity Interaction Transformer controller, vector-input port: a relational transformer
    over per-object entities, so the policy keeps clean per-object actions and values with
    many objects present. The buffer stores raw fixed-width rows ``[robot | tokens | mask]``;
    :class:`EITAgent` unpacks them into entities on the fly.

    Self-contained: the env loop, observation encoding (:class:`ObsCodec`), OU exploration,
    HER (:class:`HERRelabeler`) and the count curriculum (the env's schedule) are composed
    here."""

    def __init__(self, env, cfg: EITConfig) -> None:
        super().__init__(env=env, cfg=cfg)
        self.codec = ObsCodec(cfg.obs_pos_scale, cfg.obs_vel_scale, cfg.obs_referr_scale,
                              cfg.obs_clip)
        robot, tokens, _ = self.codec.tokens(env.observe())
        self.layout = RowLayout(robot.shape[1], tokens.shape[2], env.max_objects)
        self.action_dim = env.plant.action_dim
        self.agent = EITAgent(self.layout, self.action_dim, cfg)
        self.device = self.agent.device
        #: the rows the rollout acts from next, set by the first `learn`
        self._rows = None

    def encode(self, obs: BatchedObservation) -> torch.Tensor:
        """(B, raw_dim) buffer rows of ``obs``."""
        return self.layout.pack(*self.codec.tokens(obs))

    def act_batched(self, obs: BatchedObservation, dones=None) -> torch.Tensor:
        return self.agent.select_action(self.encode(obs))

    def learn(self, n_steps: int) -> None:
        """Train for ``n_steps`` more env steps; re-entrant (see :meth:`Algorithm.learn`)."""
        env = self.env
        cfg = self.cfg
        b = env.num_scenes
        if self._rows is None:
            self.agent.replay_buffer = LAP(self.layout.raw_dim, self.action_dim, self.device,
                                           cfg.buffer_size, batch_size=cfg.batch_size)
            if cfg.distill_teacher is None:
                obs0 = env.reset()
                self.reward.reset(obs0)
                self._rows = self.encode(obs0)
            else:
                distill_from_teacher(self)
            self._ep_reward = torch.zeros(b, device=self.device)
            self._buffer_prev_index = torch.full((b,), -1, dtype=torch.int64, device=self.device)
            self._her_count = 0
            self._data_count = 0
            # Rows written to the ring since each scene's current episode began (HER rows
            # included). HER walks the episode back through prev_index; once the ring has
            # wrapped past the episode's first row the chain is stale and can cycle forever.
            self._writes = 0
            self._ep_start = np.zeros(b, dtype=np.int64)
            self._ep_len = np.zeros(b, dtype=np.int64)
            self._noise = OUNoise(b, self.action_dim, self.device, cfg.ou_noise, cfg.ou_theta)
            self._her = HERRelabeler(self.reward, self.layout, cfg.obs_pos_scale,
                                     cfg.obs_vel_scale)

        buf = self.agent.replay_buffer
        target = self.total_steps + n_steps
        while self.total_steps < target:
            rows = self._rows
            allow_train = buf.size >= cfg.warmup_steps
            if allow_train:
                actions = self._noise(self.agent.select_action(rows))
            else:
                actions = torch.rand(b, self.action_dim, device=self.device) * 2.0 - 1.0
                if buf.size == 0:
                    print(f"-- Warmup: taking {cfg.warmup_steps} random steps before training...")

            next_obs, next_obs_true, terminated, truncated, info = env.step(actions)
            rewards = self.reward.step(next_obs_true, next_obs, terminated, truncated, info)
            discount = self.reward.discount * (~terminated).float()
            dones = terminated | truncated
            any_done = bool(dones.any())
            self._ep_reward = self._ep_reward + rewards
            self.logger.add("train/mean_reward", rewards, b)
            if any_done:
                self.logger.add("train/episode_return", self._ep_reward * dones, dones.float())
                self._ep_reward = torch.where(dones, torch.zeros_like(self._ep_reward), self._ep_reward)
                self._noise.reset(dones)

            true_rows = self.encode(next_obs_true)
            next_rows = self.encode(next_obs)
            # The true next state is stored, terminal = `terminated` only (real success), never
            # timeout: a timeout must bootstrap from the true next state.
            buffer_index = buf.add(rows, actions, true_rows, rewards, discount,
                                   self._buffer_prev_index)
            self._data_count += b
            self._writes += b
            self._ep_len += 1

            if any_done and self._her_count < self._data_count * cfg.her_ratio:
                self._relabel(dones, buffer_index)
            self.reward.refresh(next_obs)
            self._buffer_prev_index = buffer_index
            self._buffer_prev_index[dones] = -1
            if any_done:
                d = dones.cpu().numpy()
                self._ep_start[d] = self._writes
                self._ep_len[d] = 0

            if allow_train:
                self.agent.train(self.logger)

            self._rows = next_rows
            self.total_steps += b
        self.logger.add("train/buffer_size", buf.size)
        self.logger.add("explore/ou_magnitude", self._noise.ou.abs().mean())

    def _relabel(self, dones: torch.Tensor, buffer_index: torch.Tensor) -> None:
        """Hindsight-relabel the episodes that just ended into the buffer."""
        buf = self.agent.replay_buffer
        dones_host = dones.cpu().numpy()
        gamma_host = self.reward.discount.cpu().numpy()
        for i in range(self.env.num_scenes):
            if not dones_host[i]:
                continue
            if self._writes - self._ep_start[i] >= buf.max_size:
                continue                        # the episode's early rows were overwritten
            ep = buf.get_episode_trace(buffer_index[i], max_len=int(self._ep_len[i]))
            if ep is None or len(ep) <= 3:
                continue
            S, A, N, R, C = (x.cpu().numpy() for x in buf.get_sub(ep))
            rel = self._her.relabel(S, A, N, R, C, float(gamma_host[i]))
            if rel is None:
                continue
            S2, A2, N2, R2, C2 = rel
            buf.add(S2, A2, N2, R2, float(gamma_host[i]) * C2)
            self._her_count += len(S2)
            self._writes += len(S2)

    def save(self, path: str) -> None:
        torch.save({"actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=True, map_location=self.device)
        a = self.agent
        a.actor.load_state_dict(ckpt["actor"])
        a.critic.load_state_dict(ckpt["critic"])
        a.actor_target.load_state_dict(ckpt["actor"])
        a.critic_target.load_state_dict(ckpt["critic"])
