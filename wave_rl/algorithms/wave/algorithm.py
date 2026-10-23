import dataclasses
import math
from typing import NamedTuple

import torch

from .config import WaveConfig
from .runtime import WaveRuntime, nearest_first
from .warp import (InputWarp, P_DIM, WarpPolicy, categorical, component_pinned_groups,
                   side_pinned_groups, surplus_pinned_groups, warp_action)
from .selector import (N_BOOK, OBJ_DIM, TOKEN_KEYS, LocalSelector, OptionValueHead,
                       masked_argmax, normalised_entropy, pick_probabilities, scene_tokens)
from wave_rl.algorithms.crop import (CROP_CORE, N_CLEARANCE, SELF_DIM,
                                          CropBatch)


class Rollout(NamedTuple):
    """One iteration's decisions of every scene: the crops (T + 1 of them, the last for the
    bootstrap), the sampled warps and what the env paid for them."""

    cores: torch.Tensor         # (T + 1, B, CROP_CORE)
    pairs: torch.Tensor         # (T + 1, B, N, pair_dim)
    masks: torch.Tensor         # (T + 1, B, N)
    indices: torch.Tensor       # (T, B, P_DIM)
    logps: torch.Tensor         # (T, B)
    values: torch.Tensor        # (T, B)
    rewards: torch.Tensor       # (T, B)
    dones: torch.Tensor         # (T, B)
    prevs: torch.Tensor         # (T, B, P_DIM) the warp each decision could hold
    last_value: torch.Tensor    # (B,)


def clipped_surrogate(ratio: torch.Tensor, advantage: torch.Tensor,
                      clip: float) -> torch.Tensor:
    """PPO's clipped policy loss, shared by the warper and the selector."""
    return -torch.min(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()


def _where_tokens(mask: torch.Tensor, new: dict, old: dict) -> dict:
    """Row-select between two token dicts, broadcasting the (B,) mask over each field."""
    return {k: torch.where(mask.reshape(-1, *([1] * (new[k].dim() - 1))), new[k], old[k])
            for k in TOKEN_KEYS}


class WaveAlgorithm(WaveRuntime):
    """WAVE: a frozen single-object skill wrapped by a learned warping policy and a learned
    selection policy, trained jointly with PPO on the scene reward.

    The runtime it inherits does not learn. What trains here is the warp head (one
    categorical per warp parameter) and the selection head (a per-candidate scorer), each
    with its own optimizer, discount and credit assignment.
    """

    def __init__(self, skill, env, cfg: WaveConfig) -> None:
        super().__init__(skill, env, cfg)
        self.warp_policy = None
        self.warp_optim = None
        self.warp_schedule = None
        self.selector = None
        self.selector_value = None
        self.selector_optim = None
        self.selector_schedule = None
        self._sel = None
        self._trained = False
        self._collecting = False
        self._iteration = 0
        self._obs = None
        self._greedy_warp = None
        self._greedy_prev = None
        self._greedy_tick = 0
        self._pending_done = None
        self._advantage_scale = None

    def _build(self) -> None:
        if self._trained:
            return
        c = self.cfg
        env = self.env
        self.B = env.num_scenes
        obs = env.reset()
        super().act_batched(obs)
        self._obs = obs

        pinned = tuple(dict.fromkeys(
            tuple(g for g in c.disabled_warp_groups.split(",") if g)
            + side_pinned_groups(c.warp_side)
            + component_pinned_groups(c.warp_component)
            + surplus_pinned_groups(self.layout, self.action_dim, self.skill.chain is not None)))
        self.warp_policy = WarpPolicy(self.h, pinned, c.warp_head, self.crop.pair_dim)
        self.warp_policy.to(self.device)
        self.warp_optim = torch.optim.Adam(self.warp_policy.parameters(), lr=c.learning_rate)
        self.warp_schedule = self._schedule(self.warp_optim, c.ppo.learning_rate_final)
        print(f"-- WARP HEAD  P_DIM={P_DIM} bins={self.warp_policy.n_bins} "
              f"pinned={list(pinned)}")
        print(f"-- CLEARANCE  n={N_CLEARANCE} from pair rows of {self.crop.pair_dim}")
        print(f"-- RECIPE  lr={c.learning_rate:.3g}->{c.ppo.learning_rate_final:.3g} "
              f"sel={c.selector.learning_rate:.3g}->{c.selector.learning_rate_final:.3g} "
              f"(cosine) | selector={c.selector_mode} kl_penalty={c.ppo.kl_penalty_coef:.3g}")

        if c.selector_mode == "learned":
            self._selector_build()
            print(self.selector.banner())
            self._option_build(self.B, obs.active_mask.shape[1])

        self.reward.reset(obs)
        self._prev_warp = torch.zeros(self.B, P_DIM, device=self.device)
        self._trained = True

    @property
    def decision_ticks(self) -> int:
        return self.cfg.cadence_ticks

    def _guided_action(self, obs, warp, identity: bool = False) -> torch.Tensor:
        """One control tick under `warp`: the runtime's own machinery for the naive action,
        then the skill re-run under the warped presentation and its output warped too."""
        dones = self._pending_done
        self._pending_done = None
        naive = super().act_batched(obs, dones=dones)
        if identity:
            return naive
        action = self._warp_output(self._warped_input_action(obs, warp), obs, warp)
        return torch.where(self._state.has_atom.unsqueeze(1), action, naive)

    def _warped_input_action(self, obs, warp) -> torch.Tensor:
        """(B, A) the skill's action on the presentation the input side of `warp` makes."""
        return self.skill.act(self.present(obs, self._state.atom, InputWarp(warp)))

    def _warp_output(self, action, obs, warp) -> torch.Tensor:
        """(B, A) `action` under the output side of `warp`."""
        return warp_action(action, warp, self.layout, self.skill.chain, obs.robot_qpos)

    def _selector_build(self) -> None:
        """The head, its critic and their optimiser."""
        self.selector = LocalSelector(self.crop.length_scale, self.crop.length_outer, self.h,
                                      self.cfg.selector).to(self.device)
        self.selector_value = OptionValueHead(self.cfg.selector).to(self.device)
        self.selector_optim = torch.optim.Adam(
            list(self.selector.parameters()) + list(self.selector_value.parameters()),
            lr=self.cfg.selector.learning_rate)
        self.selector_schedule = self._schedule(self.selector_optim,
                                                self.cfg.selector.learning_rate_final)

    def _schedule(self, optim, final: float):
        """A cosine from the optimiser's rate down to `final` over the iterations this run
        adds (the budget is the total, a warm start's own steps included), stepped once per
        iteration. A replayed head has no budget left and gets a one-iteration cosine it
        never steps."""
        c = self.cfg
        per_iteration = c.decisions_per_iteration * c.cadence_ticks * self.env.num_scenes
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(1, math.ceil((self.budget - self.total_steps) / per_iteration)),
            eta_min=final)

    def _option_build(self, B: int, N: int) -> None:
        dev = self.device
        cap = self.cfg.selector.option_capacity
        z = lambda *s, **k: torch.zeros(*s, device=dev, **k)      # noqa: E731
        shapes = {"pos": (N, 3), "goal": (N, 3), "vel": (N, 6), "quat": (N, 4), "active": (N,),
                  "ee": (3,), "self_obs": (SELF_DIM,), "book": (N, N_BOOK),
                  "clear": (N, N_CLEARANCE)}

        def tokens(*lead):
            return {k: z(*lead, *shapes[k],
                         dtype=torch.bool if k == "active" else torch.float32)
                    for k in TOKEN_KEYS}

        self._opt_open = z(B, dtype=torch.bool)
        self._opt_tokens = tokens(B)
        self._opt_pool = z(B, N, dtype=torch.bool)
        self._opt_idx = z(B, dtype=torch.long)
        self._opt_logp = z(B)
        self._opt_eps = z(B)
        self._opt_return = z(B)
        self._opt_discount = torch.ones(B, device=dev)
        self._opt_length = z(B, dtype=torch.long)
        #: id of the chain (= attempt) the open sample belongs to; ids are unique for the
        #: life of the run, so a ring row is matched to its attempt by id alone
        self._opt_chain = z(B, dtype=torch.long)
        self._chain_next = z((), dtype=torch.long)
        self._pend_valid = z(B, dtype=torch.bool)
        self._pend_tokens = tokens(B)
        self._pend_pool = z(B, N, dtype=torch.bool)
        self._pend_idx = z(B, dtype=torch.long)
        self._pend_logp = z(B)
        self._pend_eps = z(B)
        self._pend_prev_outcome = z(B, dtype=torch.bool)
        self._ring = {
            **tokens(cap + 1),
            "pool": z(cap + 1, N, dtype=torch.bool), "idx": z(cap + 1, dtype=torch.long),
            "logp": z(cap + 1),
            "eps": z(cap + 1),
            "return": z(cap + 1), "discount": z(cap + 1), "bootstrap": z(cap + 1),
            "length": z(cap + 1, dtype=torch.long),
            "chain": torch.full((cap + 1,), -1, dtype=torch.long, device=dev),
            "target": z(cap + 1), "weight": z(cap + 1),
            "labelled": z(cap + 1, dtype=torch.bool), "used": z(cap + 1, dtype=torch.bool),
        }
        self._ring_ptr = 0
        self._ring_n = z((), dtype=torch.long)

    def _prepare_selection(self, obs, state, sat) -> None:
        """ONE selector pass per decision tick: tokens, scores and the feasibility of every
        object, including the incumbent, which is in the pool it is re-elected from."""
        if self.selector is None:
            self._sel = None
            return
        with torch.no_grad():
            B, N = obs.active_mask.shape
            every = torch.arange(N, device=self.device)[None].expand(B, -1)
            pairs, mask = self.crop.pair_features(obs, sat, every)
            clear = self.warp_policy.clearance(
                pairs.reshape(B * N, N, self.crop.pair_dim),
                mask.reshape(B * N, N)).reshape(B, N, N_CLEARANCE)
            tokens = scene_tokens(obs, state, sat, self.crop.self_state(obs), clear)
            scores, feasibility, _ = self.selector(tokens)
        self._sel = (tokens, scores, torch.sigmoid(feasibility))

    def _select(self, obs, state, pool, redo, completed, fresh) -> torch.Tensor:
        """WHICH pooled object to work next. The pool and the `redo` gate belong to the
        runtime; this only re-orders within the pool it is handed — and that pool CONTAINS
        the incumbent, so re-electing it is how a commitment continues. Until the learned
        head exists (the first tick of `_build`) the analytic rule picks."""
        has = pool.any(dim=1)
        taken = redo & has
        mode = self.cfg.selector_mode
        if mode == "learned" and self.selector is None:
            mode = "analytic"

        if mode == "analytic":
            idx = nearest_first(obs, pool)
        elif mode == "random":
            u = torch.rand(pool.shape, device=pool.device)
            idx = torch.where(pool, u, torch.full_like(u, -1.0)).argmax(dim=1)
        else:
            tokens, scores, feasibility = self._sel
            if self._collecting:
                # the coverage floor is charged ONCE PER COMMITMENT, not per decision: it
                # ignores the incumbent's stickiness, so at tens of decisions per chain a
                # constant floor would re-create by exploration the churn beta prevents.
                eps = fresh.float() * self.cfg.selector.epsilon_uniform
                p = pick_probabilities(scores, pool, eps)
                p = torch.where(has.unsqueeze(1), p, torch.full_like(p, 1.0 / p.shape[1]))
                dist = categorical(p)
                idx = dist.sample()
                logp = dist.log_prob(idx)
                self._record_pick(tokens, pool, idx, logp, eps, taken, completed)
            else:
                idx = masked_argmax(scores, pool)
        return torch.where(taken, idx, state.atom)

    def _record_pick(self, tokens, pool, idx, logp, eps, taken, prev_outcome) -> None:
        """Stash this window's pick, branch-free. Promoted to the open sample next window.
        `prev_outcome` says whether the object this decision closes was completed. `eps` is
        the exploration floor this row actually sampled under, and it is stored because the
        importance ratio must be taken against the behaviour distribution that was used —
        which differs between chain starts and continuations."""
        m1, m2 = taken, taken.unsqueeze(1)
        self._pend_valid = self._pend_valid | taken
        self._pend_tokens = _where_tokens(taken, tokens, self._pend_tokens)
        self._pend_pool = torch.where(m2, pool, self._pend_pool)
        self._pend_idx = torch.where(m1, idx, self._pend_idx)
        self._pend_logp = torch.where(m1, logp, self._pend_logp)
        self._pend_eps = torch.where(m1, eps, self._pend_eps)
        self._pend_prev_outcome = torch.where(m1, prev_outcome, self._pend_prev_outcome)

    def _option_step(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        """One decision window of chain bookkeeping.

        A CHAIN is a run of consecutive decisions that re-elect the same object — one
        ATTEMPT at that object. Every decision closes one policy-gradient sample and opens
        the next, so the reward between two decisions is credited to the choice that
        spanned it, and "keep going" is a graded action rather than the absence of one.
        That is what trains the incumbent token: the object carrying
        `is_currently_selected = 1` is in the pool, is elected, and receives gradient at
        every decision — so the score the exit reads is a score that was trained, at the
        inputs it is read at.

        The feasibility head is the probability that an ATTEMPT ends in completion. Every
        decision of a chain is labelled with the outcome of the attempt it belongs to, when
        that attempt ends:

            completed                      -> 1 on every decision of the chain
            handed over / held to the end  -> 0 on every decision of the chain

        each row weighted 1/L, so an attempt of L decisions carries unit weight. A
        hand-over IS a zero: the head estimates the value of attempting under the policy
        that is actually run, and an attempt the policy abandoned did not complete; without
        that label an object that is never completed and never held to the end would have
        no label at all. Rows wait in the ring, unlabelled, until their chain closes; the
        BCE reads closed, not yet consumed rows only.

        An episode boundary closes the open sample as TERMINAL (bootstrap 0) and DISCARDS
        any pick recorded in the same window: after the auto-reset that pick belongs to a
        different scene.
        """
        g = self.cfg.selector.gamma
        picked = self._pend_valid & ~done
        with torch.no_grad():
            _, _, local_next = self.selector(self._pend_tokens)
            v_next = self.selector_value(local_next[..., :OBJ_DIM],
                                         self._pend_tokens["self_obs"], self._pend_pool)
        zero = torch.zeros_like(v_next)

        terminal = self._opt_open & done
        self._opt_return = self._opt_return + self._opt_discount * reward * terminal.float()
        self._opt_length = self._opt_length + terminal.long()
        self._option_push(terminal, zero)

        closing = self._opt_open & picked
        finished = closing & self._pend_prev_outcome
        same = closing & ~finished & (self._pend_idx == self._opt_idx)
        self._option_push(closing, v_next)

        ends = terminal | (closing & ~same)
        self._label_chains(ends, finished)

        starts = picked & ~same
        self._opt_chain = torch.where(
            starts, self._chain_next + torch.cumsum(starts.long(), dim=0) - 1,
            self._opt_chain)
        self._chain_next = self._chain_next + starts.sum()

        m1, m2 = picked, picked.unsqueeze(1)
        self._opt_tokens = _where_tokens(picked, self._pend_tokens, self._opt_tokens)
        self._opt_pool = torch.where(m2, self._pend_pool, self._opt_pool)
        self._opt_idx = torch.where(m1, self._pend_idx, self._opt_idx)
        self._opt_logp = torch.where(m1, self._pend_logp, self._opt_logp)
        self._opt_eps = torch.where(m1, self._pend_eps, self._opt_eps)
        self._opt_return = torch.where(m1, torch.zeros_like(self._opt_return), self._opt_return)
        self._opt_discount = torch.where(m1, torch.ones_like(self._opt_discount),
                                         self._opt_discount)
        self._opt_length = torch.where(m1, torch.zeros_like(self._opt_length), self._opt_length)
        self._opt_open = (self._opt_open | picked) & ~done

        live = self._opt_open
        self._opt_return = self._opt_return + self._opt_discount * reward * live.float()
        self._opt_discount = torch.where(live, self._opt_discount * g, self._opt_discount)
        self._opt_length = self._opt_length + live.long()
        self._pend_valid.zero_()

    def _label_chains(self, ends: torch.Tensor, finished: torch.Tensor) -> None:
        """Write the outcome of every chain that closed this window onto ALL of its rows
        still in the ring: target 1[completed], weight 1/L over the L rows found. Sorted
        ids plus one searchsorted over the ring — no host sync, no per-chain loop."""
        cap = self.cfg.selector.option_capacity
        r = self._ring
        ids, order = torch.sort(torch.where(ends, self._opt_chain,
                                            torch.full_like(self._opt_chain, -1)))
        y = finished[order].float()
        chain = r["chain"][:cap]
        pos = torch.searchsorted(ids, chain).clamp_max(ids.shape[0] - 1)
        hit = (ids[pos] == chain) & (chain >= 0)
        rows = torch.zeros_like(y).scatter_add_(0, pos, hit.float())
        r["labelled"][:cap] = r["labelled"][:cap] | hit
        r["target"][:cap] = torch.where(hit, y[pos], r["target"][:cap])
        r["weight"][:cap] = torch.where(hit, 1.0 / rows[pos].clamp_min(1.0), r["weight"][:cap])

    def _option_push(self, mask: torch.Tensor, bootstrap: torch.Tensor) -> None:
        """Append the closing samples to the ring. Rows outside the mask are addressed to
        the dump slot, which is never read — one scatter, no host sync."""
        cap = self.cfg.selector.option_capacity
        k = torch.cumsum(mask.long(), dim=0) - 1
        pos = torch.where(mask, (self._ring_ptr + k) % cap, torch.full_like(k, cap))
        r = self._ring
        for key in TOKEN_KEYS:
            r[key][pos] = self._opt_tokens[key]
        r["pool"][pos] = self._opt_pool
        r["idx"][pos] = self._opt_idx
        r["logp"][pos] = self._opt_logp
        r["eps"][pos] = self._opt_eps
        r["return"][pos] = self._opt_return
        r["discount"][pos] = self._opt_discount
        r["bootstrap"][pos] = bootstrap
        r["length"][pos] = self._opt_length
        r["chain"][pos] = self._opt_chain
        # a fresh row is unlabelled until its chain closes (`_label_chains`)
        r["target"][pos] = torch.zeros_like(self._opt_return)
        r["weight"][pos] = torch.zeros_like(self._opt_return)
        r["labelled"][pos] = torch.zeros_like(mask)
        r["used"][pos] = torch.zeros_like(mask)
        n = mask.sum()
        self._ring_ptr = int((self._ring_ptr + int(n)) % cap)
        self._ring_n = self._ring_n + n

    def _selector_update(self) -> dict:
        """The selection policy's own optimisation pass, over closed options.

        A separate loop rather than a term inside the warp minibatch BECAUSE THE TWO DO NOT
        SHARE AN INDEX SPACE: the warp's sample is a decision window of the current
        rollout, the selector's is an option that may have opened four rollouts ago.

        Three losses on the one trunk:
          policy      clipped PG on the one-step OPTION TD `A = R + gamma_sel^L V(s') - V(s)`
                      (no cross-option GAE: the horizon beyond the next hand-over lives in V
                      through gamma_sel, which keeps the target bounded and count-free)
          feasibility BCE of sigma(h_pick) against the OUTCOME OF THE ATTEMPT the decision
                      belongs to: 1 if that chain ended in completion, 0 if it was handed
                      over or held to the episode end, weight 1/L per row so every attempt
                      counts once. Read from the ring's closed, not yet consumed rows —
                      whichever iteration pushed them — so no attempt is lost to the
                      rollout boundary.
          value       MSE of the pooled-quantile critic
        plus the pool-normalised entropy regulariser.
        """
        sc = self.cfg.selector
        cap = sc.option_capacity
        n = int(self._ring_n.clamp_max(cap))
        if n < 8:
            return {}
        dev = self.device
        ids = (self._ring_ptr - n + torch.arange(n, device=dev)) % cap
        r = self._ring
        tokens = {k: r[k][ids] for k in TOKEN_KEYS}
        pool, idx = r["pool"][ids], r["idx"][ids]
        logp_old = r["logp"][ids]
        eps = r["eps"][ids]
        target = r["return"][ids] + r["discount"][ids] * r["bootstrap"][ids]
        # the feasibility set: every closed attempt's rows not yet trained on
        lab_ids = (r["labelled"][:cap] & ~r["used"][:cap]).nonzero().squeeze(1)
        n_lab = int(lab_ids.shape[0])
        lab_tokens = {k: r[k][lab_ids] for k in TOKEN_KEYS}
        lab_idx, lab_y, lab_w = r["idx"][lab_ids], r["target"][lab_ids], r["weight"][lab_ids]
        with torch.no_grad():
            u_all, _ = self.selector.objects(tokens)
            advantage = target - self.selector_value(u_all, tokens["self_obs"], pool)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        acc = torch.zeros(5, dtype=torch.float64, device=dev)
        batches = 0
        size = sc.minibatch_size
        clip = sc.clip_ratio
        for _ in range(sc.update_epochs):
            perm = torch.randperm(n, device=dev)
            perm_lab = torch.randperm(max(n_lab, 1), device=dev)
            for j, start in enumerate(range(0, n, size)):
                mb = perm[start:start + size]
                t = {k: tokens[k][mb] for k in TOKEN_KEYS}
                p_mask, picked = pool[mb], idx[mb]
                scores, _, local = self.selector(t)
                dist = categorical(pick_probabilities(scores, p_mask, eps[mb]))
                logp, entropy = dist.log_prob(picked), dist.entropy()
                ratio = (logp - logp_old[mb]).clamp(-20.0, 20.0).exp()
                policy_loss = clipped_surrogate(ratio, advantage[mb], clip)
                value = self.selector_value(local[..., :OBJ_DIM], t["self_obs"], p_mask)
                value_loss = torch.nn.functional.mse_loss(value, target[mb])
                bce = torch.zeros((), device=dev)
                if n_lab:
                    # the feasibility minibatch cycles through the closed attempts
                    ml = perm_lab[(torch.arange(size, device=dev) + j * size) % n_lab]
                    tl = {k: lab_tokens[k][ml] for k in TOKEN_KEYS}
                    _, h_lab, _ = self.selector(tl)
                    h_pick = h_lab[torch.arange(len(ml), device=dev), lab_idx[ml]]
                    y, w = lab_y[ml], lab_w[ml]
                    bce = (torch.nn.functional.binary_cross_entropy_with_logits(
                        h_pick, y, reduction="none") * w).sum() / w.sum().clamp_min(1e-6)
                regulariser = -normalised_entropy(entropy, p_mask.sum(dim=1)).mean()
                loss = (policy_loss + sc.value_coef * value_loss
                        + sc.feasibility_coef * bce
                        + sc.entropy_coef * regulariser)
                self.selector_optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p_ for g_ in self.selector_optim.param_groups for p_ in g_["params"]],
                    sc.max_grad_norm)
                self.selector_optim.step()

                with torch.no_grad():
                    kl = (logp_old[mb] - logp).mean()
                    acc += torch.stack([policy_loss.detach(), value_loss.detach(),
                                        entropy.mean().detach(), kl, bce.detach()]).double()
                batches += 1

        pl, vl, ent, kl, bce_m = (acc / batches).tolist()
        r["used"][lab_ids] = True
        out = {
            "selector/policy_loss": pl, "selector/value_loss": vl,
            "selector/pick_entropy_nats": ent, "selector/approx_kl_old_new": kl,
            "selector/feasibility_bce": bce_m,
            "selector/learning_rate": self.selector_optim.param_groups[0]["lr"],
        }
        self._ring_n.zero_()
        return out

    def _terminal_value(self, obs_true, state, warp) -> torch.Tensor:
        """(B,) the value bootstrapped into the reward when an episode hits its time limit,
        conditioned on this window's applied warp, the previous warp the critic reads."""
        return self.warp_policy(self.build_crop(obs_true, state), warp)[1]

    def _step_window(self, obs, warp, identity: bool):
        """Advance `cadence_ticks` env ticks under `warp` -> (obs, reward, done).

        The reward is the benchmark's own `SceneReward`, summed over the window's ticks until
        the scene's episode ends, plus the one term only a learner can supply:

            r     += gamma * V(s_final)   on time-limit truncation
        """
        env = self.env
        B = obs.active_mask.shape[0]
        self.reward.refresh(obs)
        reward = torch.zeros(B, device=self.device)
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        state = self._state

        for _ in range(self.cfg.cadence_ticks):
            window = (~done).float()
            action = self._guided_action(obs, warp, identity)
            obs, obs_true, terminated, truncated, info = env.step(action)
            self.reward.step(obs_true, obs, terminated, truncated, info)
            paid = {k: v * window for k, v in self.reward.terms.items()}
            reward += paid["shaping"]
            reward += paid["occupancy"] + paid["success"]
            reward += paid["placement"]

            failure, unachievable = info["sim_failure"], info["unachievable"]
            live = window * (~failure).float()
            finished = terminated | truncated
            bootstrap = truncated & ~terminated & ~unachievable & ~failure & (live > 0.5)
            if bool(bootstrap.any()):
                with torch.no_grad():
                    v_final = self._terminal_value(obs_true, state, warp)
                reward += self.reward.discount * v_final * bootstrap.float()

            self._pending_done = finished
            done |= finished
        return obs, reward, done

    def _check_crop(self, crop: CropBatch, obs, t: int) -> None:
        """Refuse a non-finite crop, naming where it came from: a clean observation puts the
        fault in the skill's projection geometry, a dirty one in the simulator."""
        bad_core = ~torch.isfinite(crop.core)
        bad_pair = ~torch.isfinite(crop.pairs) & crop.mask.unsqueeze(-1)
        if bool(bad_core.any() or bad_pair.any()):
            dirty = {k: int((~torch.isfinite(getattr(obs, k).float())).sum())
                     for k in ("obj_feat", "goal_feat", "obj_qpos", "robot_qpos", "ctrl_pos")}
            raise FloatingPointError(
                f"NON-FINITE CROP at iteration {self._iteration}, decision {t}: "
                f"{int((bad_core.any(1) | bad_pair.any(2).any(1)).sum())}/"
                f"{crop.core.shape[0]} rows, core columns "
                f"{bad_core.any(0).nonzero(as_tuple=True)[0].tolist()[:24]}, pair "
                f"columns {bad_pair.any(1).any(0).nonzero(as_tuple=True)[0].tolist()}"
                f". Non-finite counts in the raw observation: {dirty}. "
                f"If the observation is clean the fault is in the skill's projection "
                f"geometry; if it is dirty the simulator already had NaNs.")

    def _collect(self) -> Rollout:
        """`decisions_per_iteration` decisions of every scene under the sampled warp, the
        selector's chain bookkeeping alongside."""
        B, dev = self.B, self.device
        T = self.cfg.decisions_per_iteration
        policy = self.warp_policy
        obs = self._obs
        N = self._state.N
        cores = torch.zeros(T + 1, B, CROP_CORE, device=dev)
        pairs = torch.zeros(T + 1, B, N, self.crop.pair_dim, device=dev)
        masks = torch.zeros(T + 1, B, N, dtype=torch.bool, device=dev)
        indices = torch.zeros(T, B, P_DIM, dtype=torch.long, device=dev)
        logps = torch.zeros(T, B, device=dev)
        values = torch.zeros(T, B, device=dev)
        rewards = torch.zeros(T, B, device=dev)
        dones = torch.zeros(T, B, dtype=torch.bool, device=dev)
        prevs = torch.zeros(T, B, P_DIM, device=dev)

        self._collecting = True
        policy.eval()
        with torch.no_grad():
            for t in range(T):
                crop = self.build_crop(obs, self._state)
                self._check_crop(crop, obs, t)
                policy.update_normalizer(crop)
                cores[t], pairs[t], masks[t] = crop.core, crop.pairs, crop.mask
                prev = self._prev_warp
                idx, logp, value, warp = policy.sample(crop, prev)
                indices[t], logps[t], values[t] = idx, logp, value
                prevs[t] = prev
                obs, reward, done = self._step_window(obs, warp, identity=False)
                rewards[t], dones[t] = reward, done
                if self.selector is not None:
                    self._option_step(reward, done)
                self._prev_warp = torch.where(done.unsqueeze(1), torch.zeros_like(warp), warp)
            last = self.build_crop(obs, self._state)
            cores[T], pairs[T], masks[T] = last.core, last.pairs, last.mask
            last_value = policy(last, self._prev_warp)[1]
        self._collecting = False
        policy.train()
        self._obs = obs
        return Rollout(cores, pairs, masks, indices, logps, values, rewards, dones, prevs,
                       last_value)

    def _gae(self, rollout: Rollout) -> tuple[torch.Tensor, torch.Tensor]:
        """(advantages, returns), (T, B) each, under each scene's own per-decision discount."""
        T, B = rollout.rewards.shape
        gamma = self.reward.discount
        values = rollout.values
        with torch.no_grad():
            advantages = torch.zeros(T, B, device=self.device)
            running = torch.zeros(B, device=self.device)
            for t in reversed(range(T)):
                non_terminal = (~rollout.dones[t]).float()
                next_value = values[t + 1] if t < T - 1 else rollout.last_value
                delta = rollout.rewards[t] + gamma * next_value * non_terminal - values[t]
                running = delta + gamma * self.cfg.ppo.gae_lambda * non_terminal * running
                advantages[t] = running
            returns = advantages + values
        return advantages, returns

    def _warp_update(self, rollout: Rollout, advantages: torch.Tensor,
                     returns: torch.Tensor) -> dict:
        """The warper's PPO epochs -> the update's own diagnostics."""
        ppo = self.cfg.ppo
        T, B = rollout.rewards.shape
        N = rollout.pairs.shape[2]
        policy = self.warp_policy
        flat_core = rollout.cores[:T].reshape(T * B, CROP_CORE)
        flat_pairs = rollout.pairs[:T].reshape(T * B, N, self.crop.pair_dim)
        flat_mask = rollout.masks[:T].reshape(T * B, N)
        flat_idx = rollout.indices.reshape(T * B, P_DIM)
        flat_logp = rollout.logps.reshape(-1)
        flat_return = returns.reshape(-1)
        flat_prev = rollout.prevs.reshape(T * B, P_DIM)
        flat_adv = advantages.reshape(-1)

        batch_std = float(flat_adv.std())
        if self._advantage_scale is None:
            self._advantage_scale = batch_std
        scale = max(self._advantage_scale, batch_std)
        flat_adv = (flat_adv - flat_adv.mean()) / (scale + 1e-8)
        self._advantage_scale = max(batch_std, ppo.advantage_scale_decay * self._advantage_scale)

        clip = ppo.clip_ratio
        beta = ppo.kl_penalty_coef
        accumulator = torch.zeros(7, dtype=torch.float64, device=self.device)
        batches = 0
        for _ in range(ppo.update_epochs):
            perm = torch.randperm(T * B, device=self.device)
            for start in range(0, T * B, ppo.minibatch_size):
                mb = perm[start:start + ppo.minibatch_size]
                logp, entropy, value, p = policy.log_prob_entropy(
                    CropBatch(flat_core[mb], flat_pairs[mb], flat_mask[mb]),
                    flat_idx[mb], flat_prev[mb])
                log_ratio = logp - flat_logp[mb]
                ratio = log_ratio.exp()
                policy_loss = clipped_surrogate(ratio, flat_adv[mb], clip)
                value_loss = 0.5 * (value - flat_return[mb]).pow(2).mean()
                regulariser = (p * (p.clamp_min(1e-12).log()
                                    - policy.identity_prior_log)
                               * policy.bin_mask.float()).sum(-1).mean(-1).mean()
                # k3 estimate of KL(old || new) on the joint distribution, from the same
                # samples as the surrogate: non-negative, and its gradient is exact.
                kl_joint = (ratio - 1.0 - log_ratio).mean()
                loss = (policy_loss + ppo.value_coef * value_loss
                        + ppo.entropy_coef * regulariser + beta * kl_joint)
                self.warp_optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo.max_grad_norm)
                self.warp_optim.step()
                with torch.no_grad():
                    accumulator += torch.stack([
                        policy_loss.detach(), value_loss.detach(), entropy.mean().detach(),
                        regulariser.detach(), (flat_logp[mb] - logp).mean(),
                        ((ratio - 1).abs() > clip).float().mean(),
                        kl_joint.detach()]).double()
                batches += 1

        with torch.no_grad():
            explained = 1.0 - (flat_return - rollout.values.reshape(-1)).var() \
                / flat_return.var().clamp_min(1e-8)
        pl, vl, ent, reg, kl, clip_fraction, kl_k3 = (accumulator / batches).tolist()
        return {
            "ppo/learning_rate": self.warp_optim.param_groups[0]["lr"],
            "ppo/kl_joint_k3": kl_k3,
            "ppo/mean_decision_reward": rollout.rewards.mean(),
            "ppo/policy_loss": pl, "ppo/value_loss": vl,
            "ppo/entropy_per_decision": ent,
            "ppo/identity_prior_kl": reg,
            "ppo/approx_kl_old_new": kl, "ppo/clip_fraction": clip_fraction,
            "ppo/value_explained_variance": explained,
            "ppo/advantage_batch_std": batch_std,
            "ppo/advantage_remembered_scale": self._advantage_scale,
            "ppo/discount_mean": self.reward.discount.mean(),
        }

    def _run_iteration(self) -> None:
        rollout = self._collect()
        advantages, returns = self._gae(rollout)
        log = self._warp_update(rollout, advantages, returns)
        if self.selector is not None:
            log.update(self._selector_update())
        for key, value in log.items():
            self.logger.add(key, value)

        self.warp_schedule.step()
        if self.selector is not None:
            self.selector_schedule.step()
        self._iteration += 1
        c = self.cfg
        self.total_steps += c.decisions_per_iteration * c.cadence_ticks * self.B

    def learn(self, n_steps: int) -> None:
        """Re-entrant: whole PPO iterations until `total_steps` reaches the target."""
        self._build()
        target = self.total_steps + n_steps
        while self.total_steps < target:
            self._run_iteration()

    def act_batched(self, obs, dones=None):
        """Greedy deployment at the training cadence — what the batched eval sweep calls per
        tick. Falls back to the unwarped runtime until a head exists."""
        if self.warp_policy is None:
            return super().act_batched(obs, dones=dones)
        if self._state is None:
            super().act_batched(obs)
        if self._greedy_prev is not None:
            self._greedy_prev = torch.where(dones.unsqueeze(1),
                                            torch.zeros_like(self._greedy_prev),
                                            self._greedy_prev)
        self._decide_warp(obs)
        self._pending_done = dones
        return self._guided_action(obs, self._greedy_warp, self._greedy_identity)

    def _decide_warp(self, obs) -> None:
        """The greedy warp at the training cadence: decided afresh every `cadence_ticks`
        ticks, held in between."""
        if self._greedy_tick % self.cfg.cadence_ticks == 0:
            with torch.no_grad():
                prev = self._greedy_prev
                if prev is None:
                    prev = torch.zeros(obs.active_mask.shape[0], P_DIM, device=self.device)
                _, self._greedy_warp, _ = self.warp_policy.greedy(
                    self.build_crop(obs, self._state), prev)
                self._greedy_prev = self._greedy_warp
                self._greedy_identity = not bool(self._greedy_warp.abs().max() > 0)
        self._greedy_tick += 1

    def reset(self) -> None:
        super().reset()
        self._greedy_warp = None
        self._greedy_prev = None
        self._greedy_tick = 0
        self._pending_done = None

    def _checkpoint(self) -> dict:
        checkpoint = super()._checkpoint()
        checkpoint["warp_policy"] = self.warp_policy.state_dict()
        checkpoint["warp_config"] = dataclasses.asdict(self.cfg)
        checkpoint["disabled_groups"] = list(self.warp_policy.disabled_groups)
        checkpoint["iteration"] = self._iteration
        checkpoint["total_steps"] = self.total_steps
        if self.selector is not None:
            checkpoint["selector"] = self.selector.state_dict()
            checkpoint["selector_value"] = self.selector_value.state_dict()
        return checkpoint

    def _apply_checkpoint(self, checkpoint: dict) -> None:
        super()._apply_checkpoint(checkpoint)
        if self.warp_policy is None:
            self.warp_policy = WarpPolicy(self.h, tuple(checkpoint["disabled_groups"]),
                                          self.cfg.warp_head, self.crop.pair_dim).to(self.device)
            self.warp_optim = torch.optim.Adam(self.warp_policy.parameters(),
                                               lr=self.cfg.learning_rate)
        self.warp_policy.load_state_dict(checkpoint["warp_policy"])
        self._iteration = checkpoint["iteration"]
        self.total_steps = checkpoint["total_steps"]
        if "selector" in checkpoint:
            # An eval-only replay never ran `_build`, so the selector is built here or it
            # would be dropped in silence. The released heads carry `count_probe` weights,
            # a linear probe this selector does not have.
            if self.selector is None:
                self._selector_build()
            self.selector.load_state_dict({k: v for k, v in checkpoint["selector"].items()
                                           if not k.startswith("count_probe.")})
            self.selector_value.load_state_dict(checkpoint["selector_value"])

    def warm_start(self, path: str) -> None:
        """Continue TRAINING from a finished head.

        `load` is the eval path and marks the algorithm trained, so `_build` never runs after
        it — right when nothing will be optimised, wrong here: the rollout buffers, the world
        count and the optimisers all live in `_build`. Building first also keeps the warp
        optimiser valid, because a head whose pins match the checkpoint's is filled in place
        and its parameter objects survive; a head whose pins do NOT match raises here rather
        than training an optimiser over parameters nothing reads.
        """
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.total_steps = checkpoint["total_steps"]
        self._build()
        self._apply_checkpoint(checkpoint)

    def load(self, path: str) -> None:
        """Restore a trained head.

        An eval-only run reaches this before ``_build`` ever runs, so the head is constructed
        from the checkpoint's own pin list rather than from a config that may not name the
        same ablation — and ``_build`` is then a no-op, so nothing overwrites it."""
        self._apply_checkpoint(torch.load(path, weights_only=False,
                                          map_location=self.device))
        self._trained = True
