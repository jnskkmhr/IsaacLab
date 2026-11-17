# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Box-attached and independently sampled palm targets in the robot base frame."""

import torch

from isaaclab.envs.mdp.commands import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)


class BoxHandPoseCommand(CommandTerm):
    """Continuously transform designated box contact poses into base coordinates.

    Poses use XYZW quaternions. Hand contact frames are offsets from the rubber-hand links;
    left and right targets lie on the rear face of the box. Targets are not global goals.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.object = env.scene[cfg.object_name]
        self.hand_ids = self.robot.find_bodies(list(cfg.hand_names), preserve_order=True)[0]
        self.offset = torch.tensor(cfg.contact_offset, device=self.device).expand(self.num_envs, 2, 3)
        self.target_positions = torch.tensor(cfg.target_positions, device=self.device).expand(self.num_envs, 2, 3)
        self.target_quaternions = torch.tensor(cfg.target_quaternions, device=self.device).expand(self.num_envs, 2, 4)
        for name in ("position_error", "orientation_error"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def target_pose_w(self):
        """Return the two hand contact target poses in world coordinates."""
        return combine_frame_transforms(
            self.object.data.root_pos_w.torch[:, None, :].expand(-1, 2, -1),
            self.object.data.root_quat_w.torch[:, None, :].expand(-1, 2, -1),
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
    object_name: str = "box"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    hand_names: tuple[str, str] = ("left_rubber_hand", "right_rubber_hand")
    contact_offset: tuple[float, float, float] = (0.132, 0.0, 0.0)
    target_positions: tuple = ((-0.25, 0.2, 0.25), (-0.25, -0.2, 0.25))
    target_quaternions: tuple = ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))


class BoxPoseCommand(UniformPoseCommand):
    """Sample a wrist pose in the robot base frame and track its offset palm.

    Match the GT reach task: rotate the wrist-to-palm offset into the base frame
    at resampling time and add it to the sampled wrist position. The resulting
    palm target stays fixed in base coordinates until the next resampling.
    Quaternions use the current Isaac Lab XYZW convention.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.offset = torch.tensor(cfg.offset, device=self.device).expand(self.num_envs, -1)
        self.pose_command_b[:, 3:] = 0.0
        self.pose_command_b[:, 6] = 1.0

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        body_quat_b = quat_mul(
            quat_inv(self.robot.data.root_quat_w.torch[env_ids]),
            self.robot.data.body_quat_w.torch[env_ids, self.body_idx],
        )
        self.pose_command_b[env_ids, :3] += quat_apply(body_quat_b, self.offset[env_ids])

    def hand_pose_w(self):
        """Return the actual palm pose in world coordinates."""
        return combine_frame_transforms(
            self.robot.data.body_pos_w.torch[:, self.body_idx],
            self.robot.data.body_quat_w.torch[:, self.body_idx],
            self.offset,
        )

    def pose_b(self, position, quaternion):
        """Express a world pose in the robot base frame."""
        return subtract_frame_transforms(
            self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, position, quaternion
        )

    def _compute_error(self):
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        position, rotation = compute_pose_error(
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:], *self.hand_pose_w()
        )
        return position.norm(dim=-1), rotation.norm(dim=-1)

    def errors(self):
        """Return palm position and orientation tracking errors."""
        return self._compute_error()

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self._compute_error()
        environment_ids = self._env.scene._ALL_INDICES
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:], environment_ids=environment_ids
        )
        self.current_pose_visualizer.visualize(*self.hand_pose_w(), environment_ids=environment_ids)


@configclass
class BoxPoseCommandCfg(UniformPoseCommandCfg):
    """Cartesian wrist-pose sampling with a local palm tracking offset."""

    class_type: type = BoxPoseCommand
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Wrist-to-palm offset in the wrist's local frame [m]."""
