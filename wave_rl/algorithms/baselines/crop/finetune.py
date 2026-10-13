import copy
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_rl.td7.agent import AvgL1Norm, LAP_huber
from wave_rl.td7.buffer import LAP

from wave_rl.algorithms.crop import CROP_CORE, N_CLEARANCE, PAIR_DIM
from wave_rl.algorithms.wave.warp import PairEncoder

from .base import CropBaseline, CropBaselineConfig

#: what the crop adds to each first layer: the running-normalized core and the learned
#: clearances, psi's own encoding of the crop.
CROP_EMBED = CROP_CORE + N_CLEARANCE


@dataclass
class FineTuneCropConfig(CropBaselineConfig):
    """The fine-tune-with-crop arm: mu itself is trained on the scene reward, reading the
    crop through zero-initialized first-layer columns.

    The action is re-decided every env tick, as mu is inside WAVE, so that at step 0 the
    closed loop IS mu's; the update count per env step matches crop-TD7's 32 per two-tick
    decision."""

    cadence_ticks: int = 1
    decisions_per_iteration: int = 96
    #: twice crop-TD7's, so the buffer spans the same env-step history at twice the rate
    buffer_size: int = 2_000_000
    updates_per_decision: int = 16
    #: env steps during which only the critic trains and the actor stays exactly mu
    critic_warmup_steps: int = 10_000_000
    #: alpha in  -Q / mean|Q| + alpha * ||mu_theta(sigma, chi) - mu_frozen(sigma)||^2
    anchor_alpha: float = 0.0


class CoreNormalizer(nn.Module):
    """Running mean/var of the crop core (batched Welford), as psi normalizes it."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(CROP_CORE))
        self.register_buffer("var", torch.ones(CROP_CORE))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, core: torch.Tensor) -> None:
        batch_mean = core.mean(dim=0)
        batch_var = core.var(dim=0, unbiased=False)
        batch_n = torch.tensor(float(core.shape[0]), device=core.device)
        n = self.count + batch_n
        delta = batch_mean - self.mean
        self.mean += delta * (batch_n / n)
        self.var.copy_((self.var * self.count + batch_var * batch_n
                        + delta.pow(2) * self.count * batch_n / n) / n)
        self.count.copy_(n)

    def forward(self, core: torch.Tensor) -> torch.Tensor:
        return (core - self.mean) / (self.var.sqrt() + 1e-6)


class Inputs(NamedTuple):
    sigma: torch.Tensor     # (B, obs_dim) mu's own single-object observation
    core: torch.Tensor      # (B, CROP_CORE) normalized
    pairs: torch.Tensor     # (B, N, PAIR_DIM)
    mask: torch.Tensor      # (B, N) bool


def zero_linear(n_in: int, n_out: int) -> nn.Linear:
    layer = nn.Linear(n_in, n_out, bias=False)
    nn.init.zeros_(layer.weight)
    return layer


def trainable_copy(module: nn.Module) -> nn.Module:
    module = copy.deepcopy(module).train()
    for p in module.parameters():
        p.requires_grad_(True)
    return module


class CropActor(nn.Module):
    """mu's actor with the crop embedding as extra first-layer input columns, zero at init."""

    def __init__(self, mu: nn.Module, pair_hidden: int) -> None:
        super().__init__()
        self.mu = trainable_copy(mu)
        self.pairs = PairEncoder(pair_hidden)
        self.l0_crop = zero_linear(CROP_EMBED, mu.l0.out_features)

    def forward(self, x: Inputs, zs: torch.Tensor) -> torch.Tensor:
        chi = torch.cat([x.core, self.pairs(x.pairs, x.mask)], dim=1)
        a = AvgL1Norm(self.mu.l0(x.sigma) + self.l0_crop(chi))
        a = torch.cat([a, zs], 1)
        a = F.relu(self.mu.l1(a))
        a = F.relu(self.mu.l2(a))
        return torch.tanh(self.mu.l3(a))


class CropCritic(nn.Module):
    """mu's twin critic over (sigma, chi, a), extended the same way."""

    def __init__(self, q: nn.Module, pair_hidden: int) -> None:
        super().__init__()
        self.q = trainable_copy(q)
        self.pairs = PairEncoder(pair_hidden)
        self.q01_crop = zero_linear(CROP_EMBED, q.q01.out_features)
        self.q02_crop = zero_linear(CROP_EMBED, q.q02.out_features)

    def forward(self, x: Inputs, action, zsa, zs) -> torch.Tensor:
        q = self.q
        chi = torch.cat([x.core, self.pairs(x.pairs, x.mask)], dim=1)
        sa = torch.cat([x.sigma, action], 1)
        emb = torch.cat([zsa, zs], 1)
        q1 = torch.cat([AvgL1Norm(q.q01(sa) + self.q01_crop(chi)), emb], 1)
        q1 = q.q3(F.elu(q.q2(F.elu(q.q1(q1)))))
        q2 = torch.cat([AvgL1Norm(q.q02(sa) + self.q02_crop(chi)), emb], 1)
        q2 = q.q6(F.elu(q.q5(F.elu(q.q4(q2)))))
        return torch.cat([q1, q2], 1)


class FineTuneCrop(CropBaseline):
    """The paper's `fine-tuned mu with crop` row: the third treatment of mu next to crop-TD7
    (mu absent) and WAVE (mu frozen and warped).

    mu's SALE encoder stays frozen: it is a representation of sigma alone, and training it
    would move the actor's output during the critic warm-up that is supposed to hold it
    fixed. The frozen skill itself (`skill.agent`) is the anchor's mu_frozen.

    A transition is (s_k, a_k, s_{k+1}) with s_k read AFTER the runtime's tick k, i.e. at the
    object the runtime committed to on that tick — the same sigma mu itself is shown."""

    def __init__(self, skill, env, cfg: FineTuneCropConfig) -> None:
        super().__init__(skill, env, cfg)
        self.actor = None
        self._buffer = None
        self._s = None
        self._a = None
        self._pending = None
        self._updates = 0

    def _inputs(self, obs, state=None) -> Inputs:
        state = self._state if state is None else state
        crop = self.build_crop(obs, state)
        return Inputs(self.present(obs, state.atom).float(), self.norm(crop.core.float()),
                      crop.pairs.float(), crop.mask)

    def _flat(self, obs) -> torch.Tensor:
        """(B, obs_dim + CROP_CORE + N*(PAIR_DIM+1)) the raw inputs as one buffer row."""
        crop = self.build_crop(obs, self._state)
        return torch.cat([self.present(obs, self._state.atom).float(), crop.core.float(),
                          crop.pairs.float().flatten(1), crop.mask.float()], dim=1)

    def _unflat(self, flat: torch.Tensor) -> Inputs:
        d = self.obs_dim
        n = (flat.shape[1] - d - CROP_CORE) // (PAIR_DIM + 1)
        pairs = flat[:, d + CROP_CORE:d + CROP_CORE + n * PAIR_DIM]
        return Inputs(flat[:, :d], self.norm(flat[:, d:d + CROP_CORE]),
                      pairs.reshape(-1, n, PAIR_DIM), flat[:, -n:] > 0.5)

    def _zs(self, sigma):
        return self.skill.agent.fixed_encoder.zs(sigma)

    def _q(self, critic, x: Inputs, action, zs):
        return critic(x, action, self.skill.agent.fixed_encoder.zsa(zs, action), zs)

    def _terminal_value(self, obs_true, state, warp) -> torch.Tensor:
        x = self._inputs(obs_true, state)
        zs = self._zs(x.sigma)
        return self._q(self.critic, x, self.actor(x, zs), zs).mean(dim=1)

    def _build(self) -> None:
        super()._build()
        if self.actor is not None:
            return
        c = self.cfg
        mu = self.skill.agent
        self.norm = CoreNormalizer().to(self.device)
        hidden = self.cfg.warp_head.pair_hidden
        self.actor = CropActor(mu.actor, hidden).to(self.device)
        self.critic = CropCritic(mu.critic, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=c.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=c.learning_rate)
        self.value_clip = (-10.0, 30.0)
        self.q_max = torch.tensor(-1e8, device=self.device)
        self.q_min = torch.tensor(1e8, device=self.device)
        self.q_max_target = torch.tensor(0.0, device=self.device)
        self.q_min_target = torch.tensor(0.0, device=self.device)

    def _train_step(self, train_actor: bool) -> None:
        c = self.cfg
        log = self.logger
        idx, s, a, s2, r, discount = self._buffer.sample()
        x, x2 = self._unflat(s), self._unflat(s2)
        with torch.no_grad():
            zs2 = self._zs(x2.sigma)
            noise = (torch.randn_like(a) * 0.1).clamp(-0.2, 0.2)
            a2 = (self.actor_target(x2, zs2) + noise).clamp(-1, 1)
            q_next = self._q(self.critic_target, x2, a2, zs2).min(1, keepdim=True)[0]
            target = r + discount * q_next.clamp(self.q_min_target, self.q_max_target)
            torch.maximum(self.q_max, target.max(), out=self.q_max)
            torch.minimum(self.q_min, target.min(), out=self.q_min)
            target = target.clip(*self.value_clip)
            zs = self._zs(x.sigma)
        q = self._q(self.critic, x, a, zs)
        td = (q - target).abs()
        critic_loss = LAP_huber(td)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self._buffer.update_priority(td.max(1)[0].clamp(min=1.0).pow(0.4), idx)
        log.add("train/Q_loss", critic_loss)
        log.add("train/Q_mean", q.mean())

        self._updates += 1
        if train_actor and self._updates % 2 == 0:
            pi = self.actor(x, zs)
            q_pi = self._q(self.critic, x, pi, zs)
            with torch.no_grad():
                mu = self.skill.agent.actor(x.sigma, zs)
            anchor = F.mse_loss(pi, mu)
            actor_loss = -q_pi.mean() / q_pi.abs().mean().detach() + c.anchor_alpha * anchor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            log.add("train/pi_value", q_pi.mean())
            log.add("finetune/anchor_mse", anchor)

        if self._updates % 250 == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self._buffer.reset_max_priority()
            self.q_max_target.copy_(self.q_max)
            self.q_min_target.copy_(self.q_min)

    def _guided_action(self, obs, warp, identity: bool = False) -> torch.Tensor:
        """The runtime's tick, then the fine-tuned actor at the sigma and crop of the object
        that tick committed to."""
        self._tick(obs)
        with torch.no_grad():
            if self._collecting:
                self._s = self._flat(obs)
                self.norm.update(self._s[:, self.obs_dim:self.obs_dim + CROP_CORE])
                x = self._unflat(self._s)
            else:
                x = self._inputs(obs)
            action = self.actor(x, self._zs(x.sigma))
            if self._collecting:
                action = (action + torch.randn_like(action)
                          * self.cfg.exploration_noise).clamp(-1, 1)
                self._a = action
        return action

    def _collect(self, obs):
        if self._buffer is None:
            self._buffer = LAP(self.obs_dim + CROP_CORE
                               + obs.active_mask.shape[1] * (PAIR_DIM + 1),
                               self.action_dim, self.device, max_size=self.cfg.buffer_size,
                               batch_size=self.cfg.batch_size)
        obs, reward, done = self._step_window(obs, None, identity=True)
        if self._pending is not None:
            s, a, r, d = self._pending
            self._buffer.add(s, a, self._s, r, d)
        self._pending = (self._s, self._a, reward, self.reward.discount * (~done).float())
        self.logger.add("train/mean_decision_reward", reward.mean())
        return obs

    def _update(self, decisions: int) -> None:
        warming = self.total_steps < self.cfg.critic_warmup_steps
        for _ in range(decisions * self.cfg.updates_per_decision):
            self._train_step(train_actor=not warming)
        self.logger.add("train/critic_warmup", float(warming))
        self.logger.add("train/buffer_size", self._buffer.size)
        self.logger.add("finetune/crop_column_norm", self.actor.l0_crop.weight.norm())

    def act_batched(self, obs, dones=None):
        self._pending_done = dones
        return self._guided_action(obs, None, identity=True)

    def _checkpoint(self) -> dict:
        return {**super()._checkpoint(),
                "finetune": {"actor": self.actor.state_dict(),
                             "critic": self.critic.state_dict(),
                             "norm": self.norm.state_dict()}}

    def _apply_checkpoint(self, checkpoint: dict) -> None:
        super()._apply_checkpoint(checkpoint)
        stored = checkpoint["finetune"]
        self.actor.load_state_dict(stored["actor"])
        self.critic.load_state_dict(stored["critic"])
        self.norm.load_state_dict(stored["norm"])
        self.actor_target.load_state_dict(stored["actor"])
        self.critic_target.load_state_dict(stored["critic"])
