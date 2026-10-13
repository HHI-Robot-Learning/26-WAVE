import copy

import torch

from wave_rl.logger import Means
from wave_rl.td7.buffer import LAP

from .codec import RowLayout
from .config import EITConfig
from .networks import EITActor, EITCriticTwin


class EITAgent:
    """Self-contained TD3 agent with EIT actor/critic (twin-critic min, target-policy smoothing,
    fixed value clipping, LAP-Huber priorities, delayed actor, polyak targets, action
    regularisation). No SALE encoder: the EIT already learns a relational state representation."""

    def __init__(self, layout: RowLayout, action_dim: int, cfg: EITConfig) -> None:
        self.layout = layout
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dims = (layout.object_dim, layout.robot_dim, action_dim, cfg.embed_dim, cfg.eit_hidden,
                cfg.n_head, cfg.n_blocks)

        self.actor = EITActor(*dims).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = EITCriticTwin(*dims).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.critic_target = copy.deepcopy(self.critic)

        self.training_steps = 0
        #: allocated by the first `learn`, so an eval-only run never holds it
        self.replay_buffer: LAP | None = None

    def entities(self, rows: torch.Tensor):
        """Flat rows ``(B, raw_dim)`` -> ``(robot, tokens, goals, mask)``. A goal entity is the
        goal position in the gantry frame, ego_pos + goal_delta."""
        robot, tokens, mask = self.layout.unpack(rows)
        goals = (tokens[..., 0:3] + tokens[..., 3:6]) * mask.unsqueeze(-1)
        return robot, tokens, goals, mask

    def select_action(self, rows: torch.Tensor) -> torch.Tensor:
        """Deterministic policy action; the rollout loop adds the exploration noise."""
        with torch.no_grad():
            return self.actor(*self.entities(rows)).clamp(-1, 1)

    def train(self, logger: Means) -> None:
        for _ in range(self.cfg.utd):
            self._update(logger)

    def _update(self, logger: Means) -> None:
        self.training_steps += 1
        cfg = self.cfg
        indices, state, action, next_state, reward, discount = self.replay_buffer.sample()

        s = self.entities(state)
        s2 = self.entities(next_state)

        with torch.no_grad():
            noise = (torch.randn_like(action) * cfg.target_policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_action = (self.actor_target(*s2) + noise).clamp(-1, 1)
            Q_target = self.critic_target(*s2, next_action).min(1, keepdim=True)[0]
            # free bootstrap (standard TD3), then the whole target clamped to value_clip
            Q_target = reward + discount * Q_target
            clip_lo, clip_hi = cfg.value_clip
            Q_target = Q_target.clamp(clip_lo, clip_hi)

        Q = self.critic(*s, action)
        td = Q - Q_target
        critic_loss = (td ** 2).sum(1).mean()
        logger.add("train/Q_loss", critic_loss)
        logger.add("train/Q_mean", Q.mean())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_optimizer.step()

        priority = td.abs().max(1)[0].clamp(min=cfg.min_priority).pow(cfg.alpha)
        self.replay_buffer.update_priority(priority, indices)

        if self.training_steps % cfg.policy_freq == 0:
            a = self.actor(*s)
            Qpi = self.critic(*s, a).min(1, keepdim=True)[0]
            actor_loss = -Qpi.mean() + cfg.a_reg_coef * (a ** 2).mean()
            logger.add("train/pi_value", Qpi.mean())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
            self.actor_optimizer.step()
            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.mul_(1 - cfg.tau).add_(cfg.tau * p)
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.mul_(1 - cfg.tau).add_(cfg.tau * p)
