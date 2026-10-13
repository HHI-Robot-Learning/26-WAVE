import math

import torch
from torch import nn
from torch.nn import functional as F

from wave_rl.algorithms.crop import N_CLEARANCE, SELF_DIM
from wave_rl.algorithms.wave.config import SelectorConfig


#: the selector's ENTIRE input, as raw tensors. Everything the network reads is derived
#: from these by relative and bounded encodings inside `LocalSelector`; nothing here is a
#: support parent, a rest height or an envelope.
#:
#: `clear` is the exception to "derived here": it is the warper's OWN learned clearances for
#: each object as if that object were the one being worked -- the same eight numbers the
#: warper reads about the object it holds -- computed once per decision by its pair encoder
#: and handed over DETACHED. The selector never trains them. It is the same quantity, so
#: chooser and executor cannot disagree about how much room a candidate has.
TOKEN_KEYS = ("pos", "goal", "vel", "quat", "active", "ee", "self_obs", "book", "clear")
BOOK_NAMES = ("is_currently_selected", "in_reselection_cooldown", "was_satisfied_and_lost")
N_BOOK = len(BOOK_NAMES)
#: per-object encoding: object and goal relative to the hand, goal relative to the object,
#: linear and angular rate, orientation, bookkeeping, the compressed travel distance, and the
#: warper's clearances for that object.
OBJ_DIM = 3 + 3 + 3 + 3 + 3 + 4 + N_BOOK + 1 + N_CLEARANCE
#: one message: the neighbour and its goal relative to the anchor, its own goal offset, its
#: rate, and the compact-support weight that also fades the whole message.
MSG_DIM = 3 + 3 + 3 + 3 + 1
_VALUE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
#: the masked-out sentinel, and it MUST be -inf rather than a large negative number. The
#: attempt cost below is a cost in CARRIES, not in logits, and nothing bounds it by the
#: scale of the other terms: a finite sentinel can be BEATEN by a real score, every pool
#: entry then lands under the mask, the masked softmax takes its row max from a masked
#: column, and the pool's probabilities underflow to zero. With -inf the row max is always
#: a real candidate.
_NEG = float("-inf")


@torch.no_grad()
def scene_tokens(obs, state, sat: torch.Tensor, self_obs: torch.Tensor,
                 clear: torch.Tensor) -> dict:
    """(B, N, ...) raw kinematics and three per-object bookkeeping flags. `sat` is read for
    ONE flag only, "was satisfied and is not now"; whether an object is at its goal NOW is
    `goal - pos` in the encoding, never a hand-gathered bit.

    `is_currently_selected` marks the INCUMBENT, and under interruption semantics that object
    stays in the pool it is re-elected from -- so this flag is set on an object that receives
    policy gradient at every decision, and the score the exit reads is a score that was
    trained."""
    device = obs.obj_feat.device
    active = obs.active_mask > 0.5
    B, N = active.shape
    rows = torch.arange(B, device=device)
    is_current = torch.zeros(B, N, device=device)
    is_current[rows, state.atom] = state.has_atom.float()
    cooldown = (state.atom_cooldown > 0).float()
    was_lost = (state.atom_ever_sat & ~sat & active).float()
    return {"pos": obs.obj_feat.float(), "goal": obs.goal_feat.float(),
            "vel": obs.obj_qvel.float(), "quat": obs.obj_qpos[..., 3:7].float(),
            "active": active, "ee": obs.ee_position.float(), "self_obs": self_obs.float(),
            "book": torch.stack([is_current, cooldown, was_lost], dim=-1),
            "clear": clear.float()}


def _mlp(d_in: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.GELU()]
        d = hidden
    return nn.Sequential(*layers)


def _zero_head(hidden: int) -> nn.Linear:
    head = nn.Linear(hidden, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


class LocalSelector(nn.Module):
    """A LOCAL index selector: one shared per-object scorer whose only view of the rest of
    the scene is a learned message sum over the <= K nearest objects within R of the
    object and of its goal.

        s_i  = F tanh(f_i) - alpha * d_xy(hand, x_i) / h
                 - alpha * (|g_i - x_i| / h + 2) * (1 / sigma(h_i) - 1)
                 + beta * [i is the incumbent]
        pick = argmax_{i in pool} s_i          pool INCLUDES the incumbent

    `f` is the learned priority, bounded by F so a learned preference can reorder within a
    travel band but never invert a large distance gap; `alpha` prices travel in object
    half-widths and is FIXED, so the travel-bounded competition argument is a property of
    the code rather than of the optimiser; `beta` (softplus, learned) is the stickiness of
    the incumbent.

    THE SSP INDEX. `sigma(h)` is the learned probability that an attempt at i ends in
    completion, trained from realised outcomes. Minimising attempts is a stochastic shortest
    path whose value is a sum of 1/p, so the expected cost of getting i done from here is its
    first travel plus (1/p_i) attempts, each costing the carry from i to its goal plus one
    object width of handling — all in the travel units alpha prices: a sure object pays its
    carry once; a confidently infeasible one pays many carries. sigma enters the score
    DETACHED — the head is trained by its outcome labels, never by the policy gradient, which
    through 1/p would be unbounded.

    THE FEASIBILITY FLOOR IS THE COUNT-GENERALISATION BOUND, not a numerical guard. Every
    other per-object term here is bounded deliberately — the priority by F, the context by a
    kernel that is exactly zero past R, alpha by being fixed — because what breaks past the
    training count is the argmax's union bound over a growing candidate set, and bounding each
    term bounds the winner's curse with it. `1/sigma` is the one unbounded functional of a
    LEARNED quantity in the rule, and it amplifies the head's error by 1/sigma^2 precisely
    where the head is extrapolating. Capped at ~7 attempts the penalty is still ~18 logits at
    a typical carry, decisive against the arena's ~8-logit travel span, and the learned factor
    cannot outvote the geometry.

    At zero-init heads (sigma = 1/2) the rule is "shortest hand -> object -> goal path
    first, and keep what you have unless something is beta/alpha half-widths shorter".

    THERE IS NO EXIT RULE. The incumbent stays in the pool and is re-elected by the same
    argmax that elected it; letting go IS the argmax choosing someone else. `beta` is the
    deadband that makes that stable: a switch needs `s_j > s_i + beta`, and switching back
    needs the reverse with beta now on j, so the relative score must traverse a band of
    width 2*beta between consecutive decisions. Objects that are exactly equally attractive
    sit dead centre of that band and produce no switch at all.

    Count-invariance is structural, not trained: every encoding is relative and bounded;
    the context is a sum over at most K compact-support weights that are EXACTLY zero past
    R, so an object added beyond R of every existing object and goal cannot change any
    existing score; the object count enters only at the argmax. Never a mean, never a softmax, never an
    absolute position.
    """

    def __init__(self, length_scale: float, length_outer: float, h: float,
                 cfg: SelectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.k = cfg.context_k
        self.length_scale = float(length_scale)
        self.length_outer = float(length_outer)
        self.h = float(h)
        self.radius = cfg.context_radius_h * self.h
        self.register_buffer("alpha", torch.tensor(cfg.travel_alpha))
        # beta_0 = alpha * stickiness_h: the incumbent starts with a lead worth that many
        # object half-widths of travel, so at zero-init the rule is "nearest-hand-first, and
        # keep what you have unless something is stickiness_h nearer".
        beta = cfg.travel_alpha * cfg.stickiness_h
        self.beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(beta))))
        self.message = nn.ModuleList([_mlp(MSG_DIM, cfg.hidden, 2) for _ in range(2)])
        self.trunk = _mlp(OBJ_DIM + SELF_DIM + 2 * cfg.hidden, cfg.hidden, cfg.n_layers)
        self.priority = _zero_head(cfg.hidden)
        self.feasibility = _zero_head(cfg.hidden)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)

    def banner(self) -> str:
        return (f"-- SELECTOR  local index selector: K={self.k} per anchor (object, goal), "
                f"R={self.radius:.3f} m ({self.radius / self.h:.3g} h), L={self.length_scale:.3f} m "
                f"V={self.length_outer:.1f}, F={self.cfg.priority_bound:.2f}, "
                f"attempts cap=e^{self.cfg.feasibility_floor:.0f}, "
                f"alpha={float(self.alpha):.3f}/h FIXED, "
                f"beta0={float(self.beta.detach()):.3f} "
                f"({float(self.beta.detach() / self.alpha):.2g} h)")

    def compress(self, magnitude: torch.Tensor) -> torch.Tensor:
        """The crop's own length encoding, `log(1+m/L)/log(1+V/L)`: bounded on the scene."""
        L = self.length_scale
        return torch.log1p(magnitude / L) / math.log1p(self.length_outer / L)

    def compress_vec(self, v: torch.Tensor) -> torch.Tensor:
        n = v.norm(dim=-1, keepdim=True)
        return v * (self.compress(n) / n.clamp_min(1e-9))

    def objects(self, t: dict):
        """(u (B, N, OBJ_DIM), d (B, N)) — the per-object encoding and the travel distance
        in half-widths. No parameters: the critic reads `u` too."""
        pos, goal, vel, quat, ee, book = (t[k] for k in ("pos", "goal", "vel", "quat", "ee",
                                                         "book"))
        e = ee[:, None, :]
        d_xy = (pos[..., :2] - e[..., :2]).norm(dim=-1)
        u = torch.cat([self.compress_vec(pos - e), self.compress_vec(goal - e),
                       self.compress_vec(goal - pos), torch.tanh(vel[..., :3]),
                       torch.tanh(vel[..., 3:] / 10.0), quat, book,
                       self.compress(d_xy).unsqueeze(-1), t["clear"]], dim=-1)
        return torch.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0), d_xy / self.h

    def neighbourhood(self, anchor: torch.Tensor, t: dict):
        """(messages (B, N, k, MSG_DIM), weights (B, N, k)) — the <= K nearest OTHER active
        objects to each anchor, weighted `relu(1 - d/R)`. Distances are computed pairwise
        and elementwise, not by a size-dependent matmul kernel, so the same pair gives the
        same float at every N."""
        pos, goal, vel, active = t["pos"], t["goal"], t["vel"], t["active"]
        B, N, _ = pos.shape
        d = (pos[:, None, :, :] - anchor[:, :, None, :]).pow(2).sum(-1).sqrt()
        eye = torch.eye(N, dtype=torch.bool, device=pos.device)[None]
        valid = active[:, None, :] & ~eye
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        k = min(self.k, N)
        near_d, near_i = d.topk(k, dim=2, largest=False)
        w = (1.0 - near_d / self.radius).clamp(0.0, 1.0)
        idx = near_i.reshape(B, N * k, 1)

        def take(x):
            return torch.gather(x, 1, idx.expand(-1, -1, x.shape[-1])).reshape(B, N, k, -1)

        p_j, g_j, v_j = take(pos), take(goal), take(vel[..., :3])
        a = anchor[:, :, None, :]
        m = torch.cat([self.compress_vec(p_j - a), self.compress_vec(g_j - a),
                       self.compress_vec(p_j - g_j), torch.tanh(v_j), w.unsqueeze(-1)], dim=-1)
        return torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0), w

    def context(self, t: dict) -> torch.Tensor:
        """(B, N, 2*hidden) the learned local context: one message sum per anchor."""
        out = []
        for anchor, net in zip((t["pos"], t["goal"]), self.message):
            m, w = self.neighbourhood(anchor, t)
            out.append((net(m) * w.unsqueeze(-1)).sum(dim=2))
        return torch.cat(out, dim=-1)

    def forward(self, t: dict):
        """(scores (B, N), feasibility logits (B, N), trunk input (B, N, OBJ_DIM+2*hidden))."""
        u, d = self.objects(t)
        N = u.shape[1]
        local = torch.cat([u, self.context(t)], dim=-1)
        z = self.trunk(torch.cat(
            [local, t["self_obs"][:, None, :].expand(-1, N, -1).clamp(-5.0, 5.0)], dim=-1))
        f = self.priority(z).squeeze(-1)
        h = self.feasibility(z).squeeze(-1)
        carry = (t["goal"] - t["pos"]).norm(dim=-1) / self.h
        attempts = torch.exp(-F.logsigmoid(h.detach()).clamp_min(-self.cfg.feasibility_floor))
        s = (self.cfg.priority_bound * torch.tanh(f)
             - self.alpha * d
             - self.alpha * (carry + 2.0) * (attempts - 1.0)
             + self.beta * t["book"][..., 0])
        return s, h, local


def masked_argmax(scores: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """THE deployed decision rule. The only place the object count enters."""
    return torch.where(pool, scores, torch.full_like(scores, _NEG)).argmax(dim=1)


def pooled_quantiles(feat: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """(B, N, F) x (B, N) -> (B, 5F), count-free.

    Quantiles of the pool, NOT a sum or a mean: an order statistic at a fixed rank fraction
    is n-stable by construction, so the same scene summarised at n=3 and n=20 lands in the
    same region of the input space. This is the ONE set function in the selector and it
    lives in the CRITIC — a baseline is a property of the state, not of a candidate, so it
    cannot bias the per-candidate argmax that carries the count-invariance claim.
    """
    B, N, F_ = feat.shape
    big = torch.finfo(feat.dtype).max
    ordered, _ = torch.sort(torch.where(pool.unsqueeze(-1), feat,
                                        torch.full_like(feat, big)), dim=1)
    n = pool.sum(dim=1).clamp_min(1)
    outs = []
    for q in _VALUE_QUANTILES:
        k = ((n - 1).float() * q).round().long().clamp(0, N - 1)
        outs.append(ordered.gather(1, k[:, None, None].expand(-1, 1, F_)).squeeze(1))
    out = torch.cat(outs, dim=-1)
    return torch.where(pool.any(dim=1)[:, None], out, torch.zeros_like(out))


class OptionValueHead(nn.Module):
    """V_sel(s) — the selector's OWN baseline, over its own discount, from the pooled
    quantiles of the per-object encoding and the robot's own state. Bounded at every N by
    `gamma_sel < 1`, which is what keeps the target count-free."""

    def __init__(self, cfg: SelectorConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _mlp(OBJ_DIM * len(_VALUE_QUANTILES) + SELF_DIM, cfg.hidden, cfg.n_layers),
            nn.Linear(cfg.hidden, 1))

    def forward(self, u: torch.Tensor, self_obs: torch.Tensor,
                pool: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([pooled_quantiles(u, pool),
                                   self_obs.clamp(-5.0, 5.0)], dim=-1)).squeeze(-1)


def pick_probabilities(scores: torch.Tensor, pool: torch.Tensor,
                       epsilon: torch.Tensor) -> torch.Tensor:
    """The TRAINING behaviour distribution: a masked softmax over the pool mixed with a
    uniform floor over the pool, so every candidate keeps a permanent label share. A
    training device only — deployment is `masked_argmax`.

    `epsilon` is PER ROW, and that is load-bearing. The floor ignores the incumbent's
    stickiness, so at a cadence of tens of decisions per commitment a constant floor would
    re-create by exploration exactly the churn beta exists to prevent. It is charged once
    per COMMITMENT — non-zero only on rows whose decision starts a new chain — which is also
    the rate at which outcome labels are generated.
    """
    logits = torch.where(pool, scores, torch.full_like(scores, _NEG))
    p = torch.where(pool, torch.softmax(logits, dim=-1), torch.zeros_like(scores))
    uniform = pool.float() / pool.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    e = epsilon.unsqueeze(1)
    return (1.0 - e) * p + e * uniform


def normalised_entropy(entropy: torch.Tensor, n_pool: torch.Tensor) -> torch.Tensor:
    """H / log(n_pool) — without it a scene with more candidates pays more entropy bonus
    purely for being bigger."""
    return entropy / torch.log(n_pool.float().clamp_min(2.0))

