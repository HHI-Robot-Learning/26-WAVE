import torch

from wave_rl.sim.scene import Scene
from wave_rl.skills.base import SkillInputs, band_shift_z, carrot_project, truncate_xy
from wave_rl.skills.td7 import TD7Skill, load_skill_config


class GantryCubeGripperSkill(TD7Skill):
    """The parallel-jaw cube gripper on a three-axis prismatic carriage, presented in a LOCAL
    frame: the scene's horizontal origin moved to the hand and its vertical origin into the
    skill's own trained z band.

    Both shifts are computed from the TRUE observation, before any warp, so the warp keeps
    its world-frame meaning and this stays a pure change of frame. They are chosen
    INDEPENDENTLY: the xy shift corrects for transport distance beyond the skill's trained
    +-1.0 m workspace, the z shift for the tower-height ceiling on the trained object-z band.
    Both are coherent translations of every absolute position presented, so the goal DELTA —
    the quantity the skill actually servos on — is preserved exactly.
    """

    embodiment = "gantry_gripper"
    grip_action = -1
    ee_frame = None
    #: the skill's reliable reach: of the radii 0.3-2.0 m and no truncation at all, the one
    #: under which it steers best toward both the object to fetch and a carried object's goal.
    disc_radius = 1.0
    #: the skill's TRAINED z ranges, read off its own config pool: the min/max of the object z
    #: and of the carriage z over the pool it was fitted on.
    z_band_object: tuple[float, float] = (0.230, 1.000)
    z_band_carriage: tuple[float, float] = (-0.423, 0.600)

    def origin_shift(self, x: SkillInputs) -> torch.Tensor:
        shift = x.hand.clone()
        shift[:, 2] = band_shift_z(x.anchor[:, 2], x.hand[:, 2],
                                   self.z_band_object, self.z_band_carriage)
        return shift

    def localize(self, x: SkillInputs) -> SkillInputs:
        obj_q = x.obj_q.clone()
        if self.disc_radius:
            obj_q[:, :3] = truncate_xy(obj_q[:, :3], x.hand, self.disc_radius)
            x.obj_centre, x.goal_centre = carrot_project(
                x.obj_centre, x.goal_centre, x.hand, self.disc_radius)
        shift = self.origin_shift(x)
        robot_q = x.robot_q.clone()
        robot_q[:, :3] = robot_q[:, :3] - shift
        obj_q[:, :3] = obj_q[:, :3] - shift
        x.robot_q = robot_q
        x.obj_q = obj_q
        x.obj_centre = x.obj_centre - shift
        x.goal_centre = x.goal_centre - shift
        return x


class PandaCubeGripperSkill(TD7Skill):
    """The Franka arm with a parallel-jaw hand.

    No localisation at all: an arm's leading joints are angles, so no translation of the
    world is expressible in them, and its workspace is small enough that the skill was
    trained across the whole of it — there is no transport-distance problem to correct.
    """

    embodiment = "panda_gripper"
    grip_action = -1
    ee_frame = "panda_hand_tip"
    chain_joints = ("panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                    "panda_joint5", "panda_joint6", "panda_joint7")
    n_robot_dofs = 9


class PandaSpherePusherSkill(TD7Skill):
    """The Franka arm with a 3 cm ball on a 0.30 m rod bolted to the flange in place of the gripper.

    Seven joints, no jaws and no grip channel: the ball can only shove. Like the
    parallel-jaw arm it takes no localisation, because an arm's leading joints are angles
    and its whole workspace is inside the trained pool.
    """

    embodiment = "panda_pusher"
    grip_action = None
    ee_frame = "fing"
    n_robot_dofs = 7


SKILL_CLASSES = {cls.embodiment: cls for cls in
                 (GantryCubeGripperSkill, PandaCubeGripperSkill, PandaSpherePusherSkill)}


def load_skill(skill_path: str, scene: Scene, n_robot_qpos: int, n_robot_qvel: int,
               n_ctrl: int, action_dim: int, device) -> TD7Skill:
    """Build the frozen skill for ``scene``'s robot.

    The skill's sidecar names the embodiment it was trained on, and a skill deployed into a
    scene built for another robot is refused."""
    cfg = load_skill_config(skill_path)
    if cfg.embodiment != scene.robot:
        raise SystemExit(f"skill {skill_path} drives {cfg.embodiment!r}, but scene "
                         f"{scene.name!r} is built for {scene.robot!r}.")
    skill = SKILL_CLASSES[cfg.embodiment](cfg, skill_path, n_robot_qpos, n_robot_qvel, n_ctrl,
                                          action_dim, device)
    print(skill.banner())
    return skill
