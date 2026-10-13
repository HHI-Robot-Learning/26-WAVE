import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_rl.logger import Means
from wave_rl.td7.buffer import LAP


@dataclass
class TD7Config:
    """TD7 (Fujimoto et al., 2023) with the additions the skills were trained with: CAPS action
    smoothness, an anchor to a loaded policy, and an exploration-noise anneal."""

    learning_rate: float = 3e-4
    actor_lr: float | None = None        # null -> learning_rate
    batch_size: int = 256
    buffer_size: int = 1_000_000
    discount: float = 0.95
    target_update_rate: int = 250
    exploration_noise: float = 0.1
    #: linear anneal of the exploration noise to this value between the two env ticks below
    expl_noise_final: float | None = None
    expl_noise_anneal_start: int = 0
    expl_noise_anneal_end: int = 0
    value_clip: list[float] = field(default_factory=lambda: [0.0, 1.0])

    target_policy_noise: float = 0.1
    noise_clip: float = 0.2
    policy_freq: int = 2
    grad_clip: float | None = None

    alpha: float = 0.4
    min_priority: float = 1.0

    #: CAPS: penalise the actor's change between consecutive states and under state noise
    caps_temporal: float = 0.0
    caps_spatial: float = 0.0
    caps_sigma: float = 0.05
    caps_warmup_start: int = 0
    caps_warmup_end: int = 0

    #: after `load`, regularise the actor toward the loaded policy with this weight, decayed
    #: linearly to `anchor_lambda_min` over `anchor_decay_steps` training steps (0: constant)
    anchor_lambda: float = 0.0
    anchor_decay_steps: int = 0
    anchor_lambda_min: float = 0.0
    #: after `load`, seed the bootstrap clamp with `value_clip` instead of [0, 0]
    warm_restore_value_range: bool = True

    #: env ticks of uniform-random actions before the first update
    timesteps_before_training: int = 25_000
    #: after `load`, env ticks before the first update
    warm_min_ticks: int = 1000
    training_steps: int = 1              # gradient steps per env tick

    zs_dim: int = 256
    enc_hdim: int = 256
    critic_hdim: int = 256
    actor_hdim: int = 256


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = F.elu(self.zs1(state))
        zs = F.elu(self.zs2(zs))
        return AvgL1Norm(self.zs3(zs))

    def zsa(self, zs, action):
        zsa = F.elu(self.zsa1(torch.cat([zs, action], 1)))
        zsa = F.elu(self.zsa2(zsa))
        return self.zsa3(zsa)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)


class TD7Agent:
    def __init__(self, state_dim: int, action_dim: int, hp: TD7Config, buffer: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hp = hp

        self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim).to(self.device)
        actor_lr = hp.learning_rate if hp.actor_lr is None else hp.actor_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.learning_rate)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.learning_rate)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.anchor_actor = None
        self.anchor_encoder = None

        #: a frozen policy that is only run needs no replay memory
        self.replay_buffer = (LAP(state_dim, action_dim, self.device, max_size=hp.buffer_size,
                                  batch_size=hp.batch_size) if buffer else None)
        self.training_steps = 0

        self.max = torch.tensor(-1e8, dtype=torch.float32, device=self.device)
        self.min = torch.tensor(1e8, dtype=torch.float32, device=self.device)
        self.max_target = torch.tensor(0., dtype=torch.float32, device=self.device)
        self.min_target = torch.tensor(0., dtype=torch.float32, device=self.device)

    def modules(self) -> dict:
        return {"actor": self.actor, "critic": self.critic, "encoder": self.encoder,
                "fixed_encoder": self.fixed_encoder, "actor_target": self.actor_target,
                "critic_target": self.critic_target,
                "fixed_encoder_target": self.fixed_encoder_target}

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "fixed_encoder": self.fixed_encoder.state_dict()}

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, weights_only=True, map_location=self.device))

    def load_state_dict(self, models: dict) -> None:
        self.actor.load_state_dict(models["actor"])
        self.critic.load_state_dict(models["critic"])
        self.fixed_encoder.load_state_dict(models["fixed_encoder"])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
        self.encoder.load_state_dict(self.fixed_encoder.state_dict())

        if self.hp.warm_restore_value_range:
            lo, hi = self.hp.value_clip
            for t, v in ((self.min, lo), (self.max, hi),
                         (self.min_target, lo), (self.max_target, hi)):
                t.fill_(v)

        if self.hp.anchor_lambda > 0.:
            self.anchor_actor = copy.deepcopy(self.actor).eval()
            self.anchor_encoder = copy.deepcopy(self.fixed_encoder).eval()
            for m in (self.anchor_actor, self.anchor_encoder):
                for p in m.parameters():
                    p.requires_grad_(False)
            print(f"== policy anchor: actor regularised toward the loaded policy, "
                  f"lambda={self.hp.anchor_lambda} decaying over {self.hp.anchor_decay_steps} "
                  f"training steps to {self.hp.anchor_lambda_min}")

    def select_action(self, state, use_exploration=True):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            zs = self.fixed_encoder.zs(state)
            action = self.actor(state, zs)
            if use_exploration:
                action = action + torch.randn_like(action) * self.hp.exploration_noise
            return action.clamp(-1, 1)

    def evaluate_critic(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            zs = self.fixed_encoder.zs(state)
            action = self.actor(state, zs).clamp(-1, 1)
            zsa = self.fixed_encoder.zsa(zs, action)
            return self.critic(state, action, zsa, zs).mean(axis=1)

    def train(self, logger: Means, multi: int = 1):
        for _ in range(multi):
            self.train_step(*self.replay_buffer.sample(), logger)

    def train_step(self, indices, state, action, next_state, reward, discount, logger):
        hp = self.hp
        self.training_steps += 1

        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)
        logger.add("train/enc_loss", encoder_loss)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        if hp.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.grad_clip)
        self.encoder_optimizer.step()

        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)
            noise = (torch.randn_like(action) * hp.target_policy_noise).clamp(
                -hp.noise_clip, hp.noise_clip)
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1, 1)
            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)
            Q_target = self.critic_target(next_state, next_action, fixed_target_zsa,
                                          fixed_target_zs).min(1, keepdim=True)[0]
            Q_target = reward + discount * Q_target.clamp(self.min_target, self.max_target)
            torch.maximum(self.max, Q_target.max(), out=self.max)
            torch.minimum(self.min, Q_target.min(), out=self.min)

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

            clip_lo, clip_hi = hp.value_clip
            Q_target = Q_target.clip(clip_lo, clip_hi)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)
        td_loss = (Q - Q_target).abs()
        critic_loss = LAP_huber(td_loss)
        logger.add("train/Q_loss", critic_loss)
        logger.add("train/Q_mean", Q.mean())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if hp.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), hp.grad_clip)
        self.critic_optimizer.step()

        priority = td_loss.max(1)[0].clamp(min=hp.min_priority).pow(hp.alpha)
        self.replay_buffer.update_priority(priority, indices)

        if self.training_steps % hp.policy_freq == 0:
            actor = self.actor(state, fixed_zs)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
            Q = self.critic(state, actor, fixed_zsa, fixed_zs)
            logger.add("train/pi_value", Q.mean())
            actor_loss = -Q.mean()

            lam_t, lam_s = hp.caps_temporal, hp.caps_spatial
            if hp.caps_warmup_end > hp.caps_warmup_start:
                ramp = min(1.0, max(0.0, (self.training_steps - hp.caps_warmup_start)
                                    / float(hp.caps_warmup_end - hp.caps_warmup_start)))
                lam_t *= ramp
                lam_s *= ramp
            if lam_t > 0.:
                with torch.no_grad():
                    fixed_zs_next = self.fixed_encoder.zs(next_state)
                caps_t = F.mse_loss(actor, self.actor(next_state, fixed_zs_next))
                actor_loss = actor_loss + lam_t * caps_t
            if lam_s > 0.:
                noisy = state + torch.randn_like(state) * hp.caps_sigma
                with torch.no_grad():
                    fixed_zs_noisy = self.fixed_encoder.zs(noisy)
                caps_s = F.mse_loss(actor, self.actor(noisy, fixed_zs_noisy))
                actor_loss = actor_loss + lam_s * caps_s

            if self.anchor_actor is not None:
                lam_a = hp.anchor_lambda
                if hp.anchor_decay_steps > 0:
                    lam_a = max(hp.anchor_lambda_min,
                                lam_a * max(0., 1. - self.training_steps / hp.anchor_decay_steps))
                if lam_a > 0.:
                    with torch.no_grad():
                        a_anchor = self.anchor_actor(state, self.anchor_encoder.zs(state))
                    anchor_mse = F.mse_loss(actor, a_anchor)
                    actor_loss = actor_loss + lam_a * Q.abs().mean().detach() * anchor_mse

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if hp.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), hp.grad_clip)
            self.actor_optimizer.step()

        if self.training_steps % hp.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())
            self.replay_buffer.reset_max_priority()
            self.max_target.copy_(self.max)
            self.min_target.copy_(self.min)
