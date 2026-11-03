# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two box-attached hand contact targets expressed in the robot base frame."""

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, subtract_frame_transforms


class BoxHandPoseCommand(CommandTerm):
    """Continuously transform designated box contact poses into base coordinates.

    Poses use XYZW quaternions. Hand contact frames are offsets from the rubber-hand links;
    left and right targets lie on the rear face of the box. Targets are not global goals.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.box = env.scene["box"]
        self.hand_ids = self.robot.find_bodies(list(cfg.hand_names), preserve_order=True)[0]
        self.offset = torch.tensor(cfg.contact_offset, device=self.device).expand(self.num_envs, 2, 3)
        self.target_positions = torch.tensor(cfg.target_positions, device=self.device).expand(self.num_envs, 2, 3)
        self.target_quaternions = torch.tensor(cfg.target_quaternions, device=self.device).expand(self.num_envs, 2, 4)
        for name in ("position_error", "orientation_error"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def target_pose_w(self):
        """Return the two hand contact target poses in world coordinates."""
        return combine_frame_transforms(
            self.box.data.root_pos_w.torch[:, None, :].expand(-1, 2, -1),
            self.box.data.root_quat_w.torch[:, None, :].expand(-1, 2, -1),
            self.target_positions,
            self.target_quaternions,
        )

    def hand_pose_w(self):
        """Return actual hand contact poses, including the hand contact offset."""
        return combine_frame_transforms(
            self.robot.data.body_pos_w.torch[:, self.hand_ids],
            self.robot.data.body_quat_w.torch[:, self.hand_ids],
            self.offset,
        )

    def pose_b(self, position, quaternion):
        return subtract_frame_transforms(
            self.robot.data.root_pos_w.torch[:, None, :].expand(-1, 2, -1),
            self.robot.data.root_quat_w.torch[:, None, :].expand(-1, 2, -1),
            position,
            quaternion,
        )

    @property
    def command(self):
        position, quaternion = self.pose_b(*self.target_pose_w())
        return torch.cat((position, quaternion), dim=-1).flatten(1)

    def errors(self):
        target_pos, target_quat = self.target_pose_w()
        pos, quat = self.hand_pose_w()
        return torch.linalg.vector_norm(pos - target_pos, dim=-1), quat_error_magnitude(quat, target_quat)

    def _update_metrics(self):
        position, orientation = self.errors()
        self.metrics["position_error"][:] = position.mean(dim=-1)
        self.metrics["orientation_error"][:] = orientation.mean(dim=-1)

    def _resample_command(self, env_ids):
        pass  # Targets are attached to the box; resetting the box changes the command.

    def _update_command(self):
        pass  # Transform live poses when read, including immediately after an environment reset.


@configclass
class BoxHandPoseCommandCfg(CommandTermCfg):
    class_type: type = BoxHandPoseCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    hand_names: tuple[str, str] = ("left_rubber_hand", "right_rubber_hand")
    contact_offset: tuple[float, float, float] = (0.132, 0.0, 0.0)
    target_positions: tuple = ((-0.25, 0.2, 0.25), (-0.25, -0.2, 0.25))
    target_quaternions: tuple = ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
