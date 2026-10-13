import math

import torch
import torch.nn.functional as F
from torch import nn

#: a goal entity is the goal's position in the gantry frame
GOAL_DIM = 3


class ParticleAttention(nn.Module):
    """Multi-head attention of the entities ``x`` over the entities ``c`` (cross) or over
    themselves (self, ``c`` None), with a boolean key-padding mask. Tensors are 4D
    ``(B, N, T, C)`` (T=1 here) to match the reference."""

    def __init__(self, embed_dim: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
        B, N, T, C = x.size()
        kv = x if c is None else c
        kvN = kv.shape[1]
        k = self.key(kv).view(B, kvN * T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(kv).view(B, kvN * T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))        # (B, nh, N*T, kvN*T)
        att = att.masked_fill(mask.unsqueeze(1).expand(-1, self.n_head, -1, -1), float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, N * T, C)
        return self.proj(y).view(B, N, T, -1)


class EITBlock(nn.Module):
    """Pre-LN transformer block (attention + MLP residual) over entities: self-attention, or
    with ``cross`` attention over the key/value entities ``c`` (e.g. goals)."""

    def __init__(self, embed_dim: int, h_dim: int, n_head: int, cross: bool = False) -> None:
        super().__init__()
        self.cross = cross
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        if cross:
            self.ln_c = nn.LayerNorm(embed_dim)
        self.attn = ParticleAttention(embed_dim, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, h_dim), nn.ReLU(True),
            nn.Linear(h_dim, h_dim), nn.ReLU(True),
            nn.Linear(h_dim, embed_dim),
        )

    def forward(self, x_in: torch.Tensor, x_mask: torch.Tensor,
                c: torch.Tensor | None = None) -> torch.Tensor:
        if self.cross:
            c = self.ln_c(c)
        x = x_in + self.attn(self.ln1(x_in), c, x_mask)
        return x + self.mlp(self.ln2(x))


def _key_padding_mask(active: torch.Tensor) -> torch.Tensor:
    """``active`` (B, N) (1=active) -> key mask (B, 1, N), True where a key is PADDED (to be
    set to -inf). Broadcast over query rows inside the attention."""
    return (active < 0.5).unsqueeze(1)


class _EITTrunk(nn.Module):
    """Shared EIT trunk: project robot->gantry entity + objects->state entities + goals->goal
    entities, run SA -> (goal) CA -> SA, returning the per-entity representation ``x``
    (B, M, 1, E). Actor and critic share the exact entity-construction code through it."""

    def __init__(self, object_dim: int, robot_dim: int, embed_dim: int, h_dim: int,
                 n_head: int, n_blocks: int) -> None:
        super().__init__()
        self.state_proj = nn.Linear(object_dim, embed_dim)
        self.goal_proj = nn.Linear(GOAL_DIM, embed_dim)
        self.gantry_proj = nn.Linear(robot_dim, embed_dim)
        self.sa1 = EITBlock(embed_dim, h_dim, n_head)
        self.ca = EITBlock(embed_dim, h_dim, n_head, cross=True)
        self.sa_rest = nn.ModuleList([EITBlock(embed_dim, h_dim, n_head) for _ in range(n_blocks - 1)])

    def build_entities(self, robot, tokens, goals, prepend=None):
        """-> (x (B, M, 1, E), c (B, N, 1, E)). ``prepend`` is an optional (B, 1, E) extra entity
        (the critic's projected action) placed FIRST in the state set."""
        gantry = self.gantry_proj(robot).unsqueeze(1)                         # (B, 1, E)
        obj = self.state_proj(tokens)                                         # (B, N, E)
        parts = [gantry, obj] if prepend is None else [prepend, gantry, obj]
        x = torch.cat(parts, dim=1).unsqueeze(2)                              # (B, M, 1, E)
        c = self.goal_proj(goals).unsqueeze(2)                                # (B, N, 1, E)
        return x, c

    def forward(self, x, c, smask, gmask):
        x = self.sa1(x, smask)
        x = self.ca(x, gmask, c)
        for blk in self.sa_rest:
            x = blk(x, smask)
        return x


class _Aggregator(nn.Module):
    """Learned-query cross-attention readout: a single output query attends over the entity set
    ``x`` (masked), giving a permutation-invariant pooled vector. Optionally also reads out an
    extra processed entity (the critic's action entity) by adding it as a second query."""

    def __init__(self, embed_dim: int, h_dim: int, n_head: int) -> None:
        super().__init__()
        self.pool = EITBlock(embed_dim, h_dim, n_head, cross=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.linear_out = nn.Linear(embed_dim, embed_dim)
        self.out_query = nn.Parameter(0.02 * torch.randn(1, 1, embed_dim))

    def forward(self, x, smask, extra_query=None):
        B = x.shape[0]
        q = self.out_query.repeat(B, 1, 1)                                    # (B, 1, E)
        if extra_query is not None:
            q = torch.cat([q, extra_query], dim=1)                            # (B, 2, E)
        agg = self.pool(q.unsqueeze(2), smask, x).squeeze(2)                  # (B, Q, E)
        return self.linear_out(self.ln(agg))                                  # (B, Q, E)


class EITActor(nn.Module):
    """EIT policy: gantry+state entities, SA -> (goal) CA -> SA, aggregate with a learned output
    query, MLP -> tanh(action). Permutation-invariant, count-generalising.

    The output layer is small-initialised so the initial action is ~0 (unsaturated): a large
    pre-tanh output saturates tanh from step 0, ``da/dtheta`` collapses and the actor cannot
    move regardless of the critic signal."""

    def __init__(self, object_dim: int, robot_dim: int, action_dim: int, embed_dim: int,
                 h_dim: int, n_head: int, n_blocks: int) -> None:
        super().__init__()
        self.trunk = _EITTrunk(object_dim, robot_dim, embed_dim, h_dim, n_head, n_blocks)
        self.agg = _Aggregator(embed_dim, h_dim, n_head)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, h_dim), nn.ReLU(True),
            nn.Linear(h_dim, h_dim), nn.ReLU(True),
            nn.Linear(h_dim, action_dim),
        )
        nn.init.uniform_(self.head[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, robot, tokens, goals, active):
        B = tokens.shape[0]
        gantry_active = active.new_ones((B, 1))
        smask = _key_padding_mask(torch.cat([gantry_active, active], dim=1))
        gmask = _key_padding_mask(active)
        x, c = self.trunk.build_entities(robot, tokens, goals)
        x = self.trunk(x, c, smask, gmask)
        agg = self.agg(x, smask).reshape(B, -1)                               # (B, E)
        return torch.tanh(self.head(agg))


class _EITCriticNet(nn.Module):
    """One EIT Q-network. The action is projected to its own entity (paper's key ablation),
    prepended to [gantry | objects]; the readout concatenates the aggregation-query output with
    the processed action entity, then MLP -> Q."""

    def __init__(self, object_dim, robot_dim, action_dim, embed_dim, h_dim, n_head, n_blocks):
        super().__init__()
        self.trunk = _EITTrunk(object_dim, robot_dim, embed_dim, h_dim, n_head, n_blocks)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, h_dim), nn.ReLU(True), nn.Linear(h_dim, embed_dim)
        )
        self.agg = _Aggregator(embed_dim, h_dim, n_head)
        # head input: [aggregated entity | action-entity readout | raw action]. The action,
        # buried as one maskable entity behind 3 attention blocks, barely moves Q at a low
        # UTD, so dQ/da -> 0 and the actor gets no gradient; the direct linear action path
        # guarantees dQ/da.
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim + action_dim, h_dim), nn.ReLU(True),
            nn.Linear(h_dim, h_dim), nn.ReLU(True),
            nn.Linear(h_dim, 1),
        )
        nn.init.uniform_(self.head[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, robot, tokens, goals, active, action):
        B = tokens.shape[0]
        ones = active.new_ones((B, 1))
        # set order: [action, gantry, objects] -> two always-active leading entities
        smask = _key_padding_mask(torch.cat([ones, ones, active], dim=1))
        gmask = _key_padding_mask(active)
        a = self.action_proj(action).unsqueeze(1)                             # (B, 1, E)
        x, c = self.trunk.build_entities(robot, tokens, goals, prepend=a)
        x = self.trunk(x, c, smask, gmask)
        a_out = x[:, 0]                                                       # processed action entity (B, 1, E)
        out = self.agg(x, smask, extra_query=a_out)                           # (B, 2, E)
        return self.head(torch.cat([out[:, 0], out[:, 1], action], dim=-1))  # (B, 1)


class EITCriticTwin(nn.Module):
    """Twin EIT Q-networks for the TD3-style ``min``. Fully independent (no shared weights)."""

    def __init__(self, object_dim, robot_dim, action_dim, embed_dim, h_dim, n_head, n_blocks) -> None:
        super().__init__()
        self.q1 = _EITCriticNet(object_dim, robot_dim, action_dim, embed_dim, h_dim, n_head, n_blocks)
        self.q2 = _EITCriticNet(object_dim, robot_dim, action_dim, embed_dim, h_dim, n_head, n_blocks)

    def forward(self, robot, tokens, goals, active, action):
        return torch.cat([
            self.q1(robot, tokens, goals, active, action),
            self.q2(robot, tokens, goals, active, action),
        ], dim=1)                                                             # (B, 2)
