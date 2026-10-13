from pathlib import Path

import mujoco
import numpy as np
import omegaconf
import torch

from wave_rl.config import SkillConfig
from wave_rl.sim.scene import SCENES, compile_model
from wave_rl.skills.base import EmbodimentLayout, FrozenSkill, SkillInputs, quat_to_matrix
from wave_rl.td7.agent import TD7Agent

#: the rigid marker bodies under the skill scene's object whose positions, with the centre's,
#: make up the object feature the skill observes
MARKERS = ("goal1", "goal2", "goal3", "goal4")


def load_skill_config(skill_path: str) -> SkillConfig:
    """The sidecar of ``<skill>.torch``: ``<skill>.yaml``."""
    return omegaconf.OmegaConf.to_object(omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.structured(SkillConfig),
        omegaconf.OmegaConf.load(Path(skill_path).with_suffix(".yaml"))))


def skill_model(cfg: SkillConfig) -> mujoco.MjModel:
    return compile_model((SCENES / "skills" / f"{cfg.scene}.xml").read_text())


def marker_offsets(model: mujoco.MjModel) -> np.ndarray:
    """(P, 3) each marker's offset in the object's body frame."""
    return np.stack([model.body(name).pos.copy() for name in MARKERS])


def translation_run(model: mujoco.MjModel) -> tuple[int, int]:
    """How many LEADING robot dofs are a prismatic run, and how many robot dofs there are.

    The run decides which half of the perceived-EE warp is RADIAL about the worked object (a
    world position, so the radial form is what keeps it translation-invariant) and which half
    is a plain affine (an angle has no radius about a metre-valued anchor). The gantry has 3,
    an arm 0."""
    joints = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    run = 0
    for j in joints[:3]:
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE:
            break
        run += 1
    return run, len(joints)


class TD7Skill(FrozenSkill):
    """A frozen skill that is a TD7 actor: ``<skill>.torch`` plus its ``<skill>.yaml``
    sidecar."""

    def __init__(self, cfg: SkillConfig, skill_path: str, n_robot_qpos: int, n_robot_qvel: int,
                 n_ctrl: int, action_dim: int, device) -> None:
        model = skill_model(cfg)
        offsets = marker_offsets(model)
        run, n_joints = translation_run(model)
        n_dofs = self.n_robot_dofs if self.n_robot_dofs is not None else n_joints
        # A robot publishing its end effector as a scene frame has no prismatic run to warp:
        # its leading joints are angles in radians. A Cartesian carriage's leading joints
        # are the hand itself.
        ee_translation = tuple(range(run)) if self.ee_frame is None else ()
        aux = tuple(range(len(ee_translation), n_dofs))
        # A robot with no grip channel has no jaw dofs either, so every aux dof is an
        # ordinary joint. Taking the trailing two regardless would hand an arm that ends in
        # a fixed tool (a pusher ball) its wrist joints labelled as a gripper.
        gripper_qpos = aux[-2:] if self.grip_action is not None else ()
        joint_qpos = aux[:-2] if self.grip_action is not None else aux

        self.cfg = cfg
        self.path = skill_path
        n_points = offsets.shape[0]
        obs_dim = ((n_robot_qpos + 3) + (n_robot_qvel + 6) + n_ctrl + 3 * n_points
                   + 3 * (1 + n_points))

        layout = EmbodimentLayout(
            robot=self.embodiment,
            ee_translation_qpos=ee_translation,
            gripper_qpos=gripper_qpos,
            joint_qpos=joint_qpos,
            grip_action=self.grip_action,
        )
        super().__init__(
            layout=layout,
            feature_points=torch.as_tensor(offsets, dtype=torch.float32, device=device),
            obs_dim=obs_dim,
            action_dim=action_dim,
            eps_pos=cfg.eps_pos,
            device=device,
        )

        self.agent = TD7Agent(state_dim=obs_dim, action_dim=action_dim, hp=cfg.td7,
                              buffer=False)
        self.agent.load(skill_path)
        for module in self.agent.modules().values():
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    def observe(self, x: SkillInputs) -> torch.Tensor:
        """The skill's training observation (`SkillEnv.observation`), batched:

            [ o_pos | o_vel | o_err | markers | goal feature delta ]
        """
        o_pos = torch.cat([x.robot_q, x.obj_q], dim=1)[:, :-4]
        rotation = quat_to_matrix(x.obj_q[:, 3:7])
        feature = self.feature(x.obj_centre, rotation)
        goal_feature = self.feature(x.goal_centre, quat_to_matrix(x.goal_quat))
        delta = goal_feature - feature
        if x.err_gain is not None:
            triples = delta.shape[-1] // 3
            delta = delta * (1.0 + x.err_gain).repeat(1, triples) \
                + x.err_off.repeat(1, triples)
        parts = [o_pos,
                 self.cfg.vel_scale * torch.cat([x.robot_v, x.obj_v], dim=1),
                 self.cfg.referr_scale * x.servo,
                 feature[:, 3:],
                 delta]
        return torch.cat(parts, dim=1).clamp(-10.0, 10.0)

    @torch.no_grad()
    def act(self, skill_obs: torch.Tensor) -> torch.Tensor:
        return self.agent.select_action(skill_obs, use_exploration=False).float()

    def banner(self) -> str:
        L = self.layout
        return (f"== FROZEN SKILL ({self.path} -> scenes/skills/{self.cfg.scene}.xml)\n"
                f"   object feature : centre + {self.n_feature_points} marker points at "
                f"{np.round(self.feature_points.cpu().numpy(), 4).tolist()}  "
                f"(obs dim {self.obs_dim})\n"
                f"   embodiment     : {L.robot} | ee translation qpos "
                f"{list(L.ee_translation_qpos)} | grip action "
                f"{L.grip_action if L.has_grip else 'NONE (grip warp pinned to identity)'}"
                f" | joints {list(L.joint_qpos)} | jaws {list(L.gripper_qpos)}\n"
                f"   ee source      : "
                + ("ctrl_pos[:3] (prismatic run; perceived-ee warp LIVE)"
                   if L.is_cartesian else
                   f"frame {self.ee_frame!r} (arm; perceived-ee warp LIVE as a plain affine on "
                   f"the joint angles, radians on the position ladder)")
                + f"\n   disc radius    : {self.disc_radius or 'OFF'}"
                + f" | tolerance eps_pos={self.eps_pos}")
