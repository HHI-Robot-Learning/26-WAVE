from dataclasses import dataclass

import torch

from wave_rl.env.observation import BatchedObservation


@dataclass(frozen=True)
class RowLayout:
    """The fixed-width replay-buffer row ``[robot | flat tokens | mask]`` over ``max_n``
    objects. ``pack``/``unpack`` are the single definition of that layout, shared by the env
    loop, the networks and HER; ``unpack`` serves torch and numpy rows alike."""

    robot_dim: int
    object_dim: int
    max_n: int

    @property
    def raw_dim(self) -> int:
        return self.robot_dim + self.max_n * self.object_dim + self.max_n

    def unpack(self, raw):
        """``(B, raw_dim)`` -> ``(robot (B,rd), tokens (B,max_n,od), mask (B,max_n))``."""
        rd, od, mn = self.robot_dim, self.object_dim, self.max_n
        robot = raw[:, :rd]
        tokens = raw[:, rd : rd + mn * od].reshape(raw.shape[0], mn, od)
        mask = raw[:, rd + mn * od :]
        return robot, tokens, mask

    def pack(self, robot: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``(robot, tokens, mask)`` -> ``(B, raw_dim)`` rows."""
        return torch.cat([robot, tokens.reshape(robot.shape[0], -1), mask], dim=1)


class ObsCodec:
    """The observation encoding of the EIT.

    Token per object: ``[ego_pos | goal_delta | lin_vel | satisfied]`` (positions
    relative to the robot, all scaled). The robot block is
    ``[qpos | vel*scale | ctrl_err*scale]``. Both are clipped to ``clip`` so a far object or
    a large arena cannot blow the tokens out of distribution."""

    def __init__(self, pos_scale: float, vel_scale: float, referr_scale: float,
                 clip: float) -> None:
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.referr_scale = referr_scale
        self.clip = clip

    def tokens(self, obs: BatchedObservation) -> tuple:
        """``(robot, tokens, mask)``; padded slots are zeroed via the mask after clipping, so
        stored rows stay clean."""
        ps, vs = self.pos_scale, self.vel_scale
        obj = obs.obj_feat.float()
        pos = (obj - obs.ctrl_pos[..., :3].float().unsqueeze(1)) * ps
        goal_delta = (obs.goal_feat.float() - obj) * ps
        vel = obs.obj_qvel[..., :3].float() * vs
        sat = obs.satisfied_mask.float().unsqueeze(-1)
        mask = obs.active_mask.float()
        tokens = torch.cat([pos, goal_delta, vel, sat], dim=-1)
        tokens = tokens.clamp(-self.clip, self.clip) * mask.unsqueeze(-1)
        robot = torch.cat([
            obs.robot_qpos.float(),
            obs.robot_qvel.float() * vs,
            (obs.ctrl_ref - obs.ctrl_pos).float() * self.referr_scale,
        ], dim=1).clamp(-self.clip, self.clip)
        return robot, tokens, mask
