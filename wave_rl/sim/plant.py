import math

import torch

from wave_rl.sim.sim import Sim


class Plant:
    """The robot's actuation on top of a :class:`Sim`: a policy action in [-1, 1] is a change
    of the joint-position reference, followed by a second-order polynomial over one control
    tick. With `grip_binary` the last channel is a jaw that only opens or closes, latched with
    `grip_hysteresis`."""

    def __init__(self, sim: Sim, cfg) -> None:
        self.sim = sim
        self.cfg = cfg
        self.device = sim.torch_device
        self.num_scenes = sim.num_scenes
        self.ctrl_indices = sim.ctrl_indices
        self.ctrl_indices_t = torch.as_tensor(self.ctrl_indices, dtype=torch.long,
                                              device=self.device)
        self.ctrl_current = self.qpos[:, self.ctrl_indices_t].clone()
        limits = sim.scene.limits
        self.ctrl_lower = torch.as_tensor(limits[0][self.ctrl_indices], dtype=torch.float32,
                                          device=self.device)
        self.ctrl_upper = torch.as_tensor(limits[1][self.ctrl_indices], dtype=torch.float32,
                                          device=self.device)
        self.action_scale = cfg.action_scale * math.sqrt(cfg.tau_step)
        self.action_dim = sim.ctrl_dim
        self._ee_ids = (torch.as_tensor(sim.body_ids([cfg.grasp_ee_point, sim.scene.ref_name]),
                                        dtype=torch.long, device=self.device)
                        if cfg.grasp_ee_point is not None else None)
        B = self.num_scenes
        self.grip_closed = torch.zeros(B, dtype=torch.bool, device=self.device)
        n = round(cfg.tau_step / sim.tau_sim)
        #: the reference's substep offsets within one control tick: 0, tau_sim, ..., tau_step.
        #: Relative to the tick, so the reference is as exact at hour ten as at second one.
        self._offsets = (torch.arange(n + 1, dtype=torch.float64, device=self.device)
                         * sim.tau_sim).float().view(-1, 1, 1)
        #: the reference the physics follows over one tick, one row per substep; written in
        #: place, because the simulator's captured graph reads it by address
        self._ctrl_buffer = torch.zeros((n + 1, B, self.action_dim), device=self.device)

    def step(self, action: torch.Tensor) -> None:
        """Apply `action` (B, action_dim) for one control tick."""
        cfg = self.cfg
        action_delta = self.action_scale * action
        if cfg.grip_binary:
            logit = action[:, -1]
            h = cfg.grip_hysteresis
            if h > 0.:
                closed = (self.grip_closed & (logit >= -h)) | (logit > h)
            else:
                closed = logit > 0.
            self.grip_closed = closed
            on, off = self._grip_targets()
            action_delta = action_delta.clone()
            action_delta[:, -1] = torch.where(closed, on, off) - self.ctrl_current[:, -1]
        # A critically damped second-order reference from the current one, reaching
        # `action_delta` away with the joint's velocity as its initial slope.
        velocity = self.qvel[:, self.ctrl_indices_t]
        lam = 2. * cfg.tau_step
        coeff = (action_delta - 2. * lam * velocity) / (2. * lam * lam)
        d = self._offsets
        self._ctrl_buffer.copy_(d * d * coeff + d * velocity + self.ctrl_current)
        self.sim.step(self._ctrl_buffer)
        self.ctrl_current = self._ctrl_buffer[-1]
        if cfg.ctrl_antiwindup:
            self.ctrl_current = torch.clamp(self.ctrl_current, self.ctrl_lower, self.ctrl_upper)

    def hand_position(self) -> torch.Tensor | None:
        """(B, 3) the `grasp_ee_point` body relative to the reference body, or None on a robot
        that names none (the gantry, whose leading qpos are the hand's world x/y/z)."""
        if self._ee_ids is None:
            return None
        xpos = self.sim.xpos_t
        return xpos[:, self._ee_ids[0]] - xpos[:, self._ee_ids[1]]

    def hand_xy(self) -> torch.Tensor:
        """(B, 2) the hand's xy in the feature frame: the named body on an arm, the leading
        prismatic joints on the gantry."""
        hand = self.hand_position()
        return self.qpos[:, 0:2] if hand is None else hand[:, :2]

    def grip_closedness(self) -> torch.Tensor:
        """(B,) how closed the jaw reference is, in [0, 1]: 1 clamped shut, 0 wide open."""
        frac = ((self.ctrl_current[:, -1] - self.ctrl_lower[-1])
                / (self.ctrl_upper[-1] - self.ctrl_lower[-1] + 1e-9)).clamp(0., 1.)
        return frac if self.cfg.grip_closed_at_upper else (1. - frac)

    def _grip_targets(self):
        """(reference for closed, reference for open) of the jaw channel."""
        if self.cfg.grip_closed_at_upper:
            return self.ctrl_upper[-1], self.ctrl_lower[-1]
        return self.ctrl_lower[-1], self.ctrl_upper[-1]

    @property
    def qpos(self) -> torch.Tensor:
        return self.sim.qpos_t

    @property
    def qvel(self) -> torch.Tensor:
        return self.sim.qvel_t

    @property
    def act(self) -> torch.Tensor:
        return self.sim.act_t
