from dataclasses import dataclass

import torch


@dataclass
class BatchedObservation:
    """Raw, structured world state for ``B`` scenes at once.

    A lifting policy selects an object and folds its token into the vector its frozen skill
    expects; an end-to-end set/slot policy encodes the whole per-object token set
    permutation-invariantly. Fields are raw and unscaled so each algorithm applies its own
    encoding.

    Every field is a ``torch.Tensor`` on the sim's device, and the per-object axis is padded
    to ``N = max_objects`` with inactive slots masked rather than sliced.
    """

    robot_qpos: torch.Tensor        # (B, n_robot_q)   robot joint positions
    robot_qvel: torch.Tensor        # (B, n_robot_v)   robot joint velocities
    obj_qpos: torch.Tensor          # (B, N, 7)        per-object free-joint pose
    obj_qvel: torch.Tensor          # (B, N, 6)        per-object free-joint velocity
    obj_feat: torch.Tensor          # (B, N, 3)        achieved feature (ref-relative)
    goal_feat: torch.Tensor         # (B, N, 3)        target feature (ref-relative)
    ctrl_ref: torch.Tensor          # (B, n_ctrl)      current control reference
    ctrl_pos: torch.Tensor          # (B, n_ctrl)      controlled-joint positions
    active_mask: torch.Tensor       # (B, N) bool      object physically present this episode
    satisfied_mask: torch.Tensor    # (B, N) bool      object's goal currently satisfied
    ee_pos: torch.Tensor | None = None
    ee_vel: torch.Tensor | None = None
    #: the plant's `arm_proxies` boxes, K of them: centres (B, K, 3) in the `obj_feat` frame,
    #: orientations (B, K, 3, 3) whose columns are the box axes, half sizes (K, 3). None on a
    #: plant that names no proxies.
    arm_box_pos: torch.Tensor | None = None
    arm_box_rot: torch.Tensor | None = None
    arm_box_half: torch.Tensor | None = None

    @property
    def ee_position(self) -> torch.Tensor:
        """Where the end effector is, in the frame ``obj_feat``/``goal_feat`` already use.

        Every consumer must go through this instead of slicing ``ctrl_pos``. On the gantry
        the two are the same tensor (joints 1-3 are prismatic world x/y/z); on an arm
        ``ctrl_pos[:3]`` are joint angles in radians and only the published ``ee_pos`` means
        anything.

        The two embodiments disagree in z: on the gantry this is the carriage joint, which
        sits ~0.484 m above the jaws, so ``ee_position[2]`` is not "the height of the hand"
        on the gantry; on an arm ``ee_pos`` is the true frame between the pads."""
        return self.ctrl_pos[..., :3] if self.ee_pos is None else self.ee_pos

    @property
    def ee_velocity(self) -> torch.Tensor:
        """The end effector's linear velocity, same frame and same contract as
        :attr:`ee_position`. Gantry fallback: the leading prismatic run of ``robot_qvel``."""
        return self.robot_qvel[..., :3] if self.ee_vel is None else self.ee_vel
