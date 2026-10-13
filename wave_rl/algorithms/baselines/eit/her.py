import numpy as np
import torch

from wave_rl.reward import SceneReward

from .codec import RowLayout


class HERRelabeler:
    """Hindsight Experience Replay for the fixed-width token rows ``[robot | tokens | mask]``.
    Relabels a finished episode's goals to an achieved configuration and recomputes the reward —
    the key exploration tool for sparse multi-object goal-reaching.

    The relabelled reward is the learner's own :class:`~wave_rl.reward.SceneReward`, recomputed
    from what the rows still carry: its closeness potential, occupancy price, success bonus and
    reaching term, under the episode's own discount. Satisfaction is the position (and rest)
    test the rows allow, without the live reward's debounce, and the tool-proximity term —
    which needs the hand — is not in the rows. Token layout per object:
    ``[ego_pos 0:3 | goal_delta 3:6 (= (goal-obj)*ps) | vel 6:9 | satisfied 9]``."""

    def __init__(self, reward: SceneReward, layout: RowLayout, pos_scale: float,
                 vel_scale: float) -> None:
        self.reward = reward
        self.layout = layout
        self.eps_pos = reward.measure.goal.eps_pos
        self.eps_vel = reward.measure.goal.eps_vel
        self.ps = pos_scale
        self.vs = vel_scale

    def relabel(self, S, A, N, R, C, gamma: float):
        """Relabel one episode (numpy rows) to an achieved config; returns ``(S, A, N, R, C)`` with
        ``C = continue`` (the caller stores ``gamma * C`` as the bootstrap discount), or ``None``
        when there is no usable hindsight goal. ``gamma`` is the episode's discount."""
        rd, od, mn = self.layout.robot_dim, self.layout.object_dim, self.layout.max_n
        ps, vs = self.ps, self.vs
        _, tokS, maskS = self.layout.unpack(S)
        tokS, tokN = tokS.copy(), self.layout.unpack(N)[1].copy()
        mask = maskS[0] > 0.5
        n_active = int(mask.sum())
        if n_active == 0:
            return None
        speedN = np.linalg.norm(tokN[:, :, 6:9] / vs, axis=2)         # (T, mn) m/s
        speedS = np.linalg.norm(tokS[:, :, 6:9] / vs, axis=2)
        T = S.shape[0]
        if self.eps_vel is not None:                                 # rest task: last all-at-rest step
            rest_ok = ((speedN <= self.eps_vel) | (~mask)[None, :]).all(axis=1)
            rest_t = np.where(rest_ok)[0]
            if len(rest_t) == 0 or rest_t[-1] < 3:
                return None
            sl = slice(0, int(rest_t[-1]) + 1)
        else:
            # position-only: relabel to the config achieved at a random step, then truncate the
            # episode there so the achievement is the terminal transition (objects drift on past t
            # in the real trace, which would otherwise mislabel post-t steps as terminal).
            sl = slice(0, int(np.random.randint(3, T)) + 1)          # caller guarantees T > 3
        S, A, N, R, C = S[sl], A[sl], N[sl], R[sl], C[sl]
        tokS, tokN, speedN, speedS = tokS[sl], tokN[sl], speedN[sl], speedS[sl]
        T = S.shape[0]; t = T - 1
        # (1) frame-consistent relabel via the absolute goal-delta column: new_gd(x) = orig_gd(x) -
        # orig_gd(t) = (obj_t - obj_x)*ps, with the common goal_orig cancelling.
        gd_ref = tokN[t, :, 3:6].copy()
        gdS = tokS[:, :, 3:6] - gd_ref[None]
        gdN = tokN[:, :, 3:6] - gd_ref[None]
        tokS[:, :, 3:6] = gdS
        tokN[:, :, 3:6] = gdN
        # (2) satisfaction + latched completion against the relabelled goal (dist in metres = |gd|/ps)
        distN = np.linalg.norm(gdN / ps, axis=2)
        satN = distN <= self.eps_pos
        satS = np.linalg.norm(gdS / ps, axis=2) <= self.eps_pos
        if self.eps_vel is not None:
            satN = satN & (speedN <= self.eps_vel)
            satS = satS & (speedS <= self.eps_vel)
        satN = satN & mask[None]; satS = satS & mask[None]
        comp = np.maximum.accumulate(satN, axis=0)                   # (T, mn) monotone completion
        comp_prev = np.concatenate([np.zeros((1, mn), bool), comp[:-1]], axis=0)
        newly = comp & (~comp_prev) & mask[None]                     # +1 the step each object first lands
        tokS[:, :, 9] = comp_prev.astype(np.float32)                 # observed satisfied = latched (monotone)
        tokN[:, :, 9] = comp.astype(np.float32)
        C = (comp.sum(axis=1) != n_active).astype(np.float32).reshape(-1, 1)   # continue = not all done
        # (3) the scene reward under the relabelled goal
        R = self._reward(distN, np.linalg.norm(gdS / ps, axis=2), satN, newly, mask, C, gamma)
        S, N = S.copy(), N.copy()
        S[:, rd:rd + mn * od] = tokS.reshape(T, mn * od)
        N[:, rd:rd + mn * od] = tokN.reshape(T, mn * od)
        return S, A, N, R, C

    def _reward(self, distN, distS, satN, newly, mask, C, gamma: float):
        """(T, 1) :class:`~wave_rl.reward.SceneReward`'s terms for the relabelled rows: PBRS on the
        closeness potential with terminal reclaim, occupancy income, the success bonus on the
        terminal transition and the reaching term the first time each object lands."""
        rw = self.reward
        cfg = rw.cfg
        k = rw.decision_ticks
        tick_gamma = gamma ** (1.0 / k)
        scale = cfg.occupancy_weight_scale
        n_active = int(mask.sum())
        m = mask[None].astype(np.float32)

        def phi(dist):
            close = rw.measure.closeness(torch.from_numpy(dist.astype(np.float32))).numpy()
            return cfg.phi_closeness_weight * (close * m).sum(axis=1)

        terminal = 1.0 - C[:, 0]
        phi_n = phi(distN)
        shaping = tick_gamma * phi_n - phi(distS) - tick_gamma * phi_n * terminal
        income = scale * (1.0 - gamma) / k * (satN * m).sum(axis=1) / n_active
        success = scale * gamma * terminal
        placement = scale * cfg.first_completion_bonus * newly.sum(axis=1) / n_active
        return (shaping + income + success + placement).astype(np.float32).reshape(-1, 1)
