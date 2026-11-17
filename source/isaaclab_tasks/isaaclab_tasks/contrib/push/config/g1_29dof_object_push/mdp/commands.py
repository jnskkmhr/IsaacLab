# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Box velocity commands and wrist targets on the nearest vertical box face."""

from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    subtract_frame_transforms,
    yaw_quat,
)

from isaaclab_tasks.contrib.push.mdp.commands import BoxHandPoseCommand, BoxHandPoseCommandCfg


def standing_command_mask(command: torch.Tensor, threshold: float) -> torch.Tensor:
    """Select commands with planar speed [m/s] and yaw rate [rad/s] below the threshold."""
    return (command[:, :2].norm(dim=-1) < threshold) & (command[:, 2].abs() < threshold)


def nearest_face_poses(robot_position_b, half_size, lateral_offset, top_clearance):
    """Return paired wrist poses in box coordinates; +y is left when facing into the box.

    Select the nearest of the four vertical faces by point-to-rectangle distance.
    The wrist x-axis points into that face. Dimensions and distances are in meters.
    """
    normals = robot_position_b.new_tensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])
    centers = normals * half_size
    closest = robot_position_b[:, None, :].clamp(-half_size, half_size).expand(-1, 4, -1)
    closest = torch.where(normals[None].bool(), centers[None], closest)
    face = (robot_position_b[:, None] - closest).square().sum(-1).argmin(-1)
    normal = normals[face]
    forward = -normal
    left = torch.stack((-forward[:, 1], forward[:, 0], torch.zeros_like(forward[:, 0])), dim=-1)
    width = (left.abs() * half_size).sum(-1)
    separation = width.clamp(max=lateral_offset)
    center = centers[face].clone()
    center[:, 2] = half_size[2] - top_clearance
    position = (
        center[:, None]
        + left[:, None] * separation[:, None, None] * robot_position_b.new_tensor([1, -1])[None, :, None]
    )
    yaw = torch.atan2(forward[:, 1], forward[:, 0])
    rotation = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    return position, rotation[:, None].expand(-1, 2, -1)


def project_hand_poses(position, quaternion, candidates, orientation_weight):
    """Select dataset poses by position and sign-invariant quaternion distance.

    Candidate poses have shape (hands, samples, 7) in the reference link frame.
    The orientation weight converts quaternion chord distance to meters.
    """
    selected = []
    for hand in range(position.shape[1]):
        poses = candidates[hand]
        distance = torch.cdist(position[:, hand], poses[:, :3]).square()
        alignment = (quaternion[:, hand] @ poses[:, 3:].T).square().clamp(max=1.0)
        distance += 4.0 * orientation_weight**2 * (1.0 - alignment)
        selected.append(poses[distance.argmin(dim=-1)])
    return torch.stack(selected, dim=1)


class NearestBoxFaceCommand(BoxHandPoseCommand):
    """Transform box-face wrist targets into the configured robot link frame at every read."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if any(size <= 0 for size in cfg.box_size):
            raise ValueError("Box dimensions must be positive.")
        if not 0 <= cfg.top_clearance <= cfg.box_size[2] or cfg.lateral_offset < 0:
            raise ValueError("Invalid wrist target clearance or lateral offset.")
        self.half_size = torch.tensor(cfg.box_size, device=self.device) * 0.5
        body_ids, _ = self.robot.find_bodies(cfg.reference_body_name)
        if len(body_ids) != 1:
            raise ValueError("reference_body_name must select exactly one body.")
        self.reference_body_idx = body_ids[0]
        if not 0 < cfg.contact_distance < cfg.approach_distance:
            raise ValueError("Require 0 < contact_distance < approach_distance.")
        if cfg.workspace_pose_count < 1 or cfg.orientation_weight < 0:
            raise ValueError("Require positive workspace_pose_count and nonnegative orientation_weight.")
        with np.load(cfg.dataset_path, allow_pickle=False) as data:
            if str(data["reference_body_name"].item()) != cfg.reference_body_name:
                raise ValueError("Hand workspace and command must use the same reference body.")
            poses = np.stack([data[key] for key in cfg.pose_keys])
            indices = np.linspace(0, poses.shape[1] - 1, min(cfg.workspace_pose_count, poses.shape[1]), dtype=int)
            self.workspace_poses = torch.as_tensor(poses[:, indices], device=self.device, dtype=torch.float32)
        self.approach_positions = torch.tensor(cfg.approach_positions, device=self.device)
        self.standing_poses = project_hand_poses(
            self.approach_positions[None],
            torch.tensor(cfg.standing_quaternions, device=self.device)[None],
            self.workspace_poses,
            cfg.orientation_weight,
        )

    def pose_b(self, position, quaternion):
        """Express target poses in the manipulation teacher's moving reference link."""
        return subtract_frame_transforms(
            self.robot.data.body_link_pos_w.torch[:, self.reference_body_idx, None].expand(-1, 2, -1),
            self.robot.data.body_link_quat_w.torch[:, self.reference_body_idx, None].expand(-1, 2, -1),
            position,
            quaternion,
        )

    def hand_pose_w(self):
        """Measure wrist link origins, matching the reaching teacher's body_pose_local input."""
        return combine_frame_transforms(
            self.robot.data.body_link_pos_w.torch[:, self.hand_ids],
            self.robot.data.body_link_quat_w.torch[:, self.hand_ids],
            self.offset,
        )

    def object_target_pose_w(self):
        position_b = quat_apply_inverse(
            self.object.data.root_quat_w.torch,
            self.robot.data.root_pos_w.torch - self.object.data.root_pos_w.torch,
        )
        position, rotation = nearest_face_poses(
            position_b, self.half_size, self.cfg.lateral_offset, self.cfg.top_clearance
        )
        # The teacher commands wrist origins; keep them outside the surface by
        # the configured closed-hand reach along the wrist's local +X axis.
        offset = torch.zeros_like(position)
        offset[..., 0] = self.cfg.wrist_surface_offset
        position -= quat_apply(rotation, offset)
        return combine_frame_transforms(
            self.object.data.root_pos_w.torch[:, None].expand(-1, 2, -1),
            self.object.data.root_quat_w.torch[:, None].expand(-1, 2, -1),
            position,
            rotation,
        )

    def target_pose_w(self):
        """Approach with a comfortable reach, then select feasible box-contact wrist poses."""
        position, quaternion = self.pose_b(*self.object_target_pose_w())
        distance = position.norm(dim=-1, keepdim=True)
        blend = (
            (self.cfg.approach_distance - distance) / (self.cfg.approach_distance - self.cfg.contact_distance)
        ).clamp(0.0, 1.0)
        position = torch.lerp(self.approach_positions, position, blend)
        poses = project_hand_poses(position, quaternion, self.workspace_poses, self.cfg.orientation_weight)
        standing = standing_command_mask(
            self._env.command_manager.get_command(self.cfg.velocity_command_name), self.cfg.command_threshold
        )
        # Standing wrist targets are fixed in the torso frame, independent of the box.
        poses = torch.where(standing[:, None, None], self.standing_poses, poses)
        return combine_frame_transforms(
            self.robot.data.body_link_pos_w.torch[:, self.reference_body_idx, None].expand(-1, 2, -1),
            self.robot.data.body_link_quat_w.torch[:, self.reference_body_idx, None].expand(-1, 2, -1),
            poses[..., :3],
            poses[..., 3:],
        )


@configclass
class NearestBoxFaceCommandCfg(BoxHandPoseCommandCfg):
    """Two wrist targets; dimensions must match the scene's box."""

    class_type: type = NearestBoxFaceCommand
    velocity_command_name: str = MISSING
    command_threshold: float = MISSING
    standing_quaternions: tuple = MISSING
    reference_body_name: str = MISSING
    box_size: tuple[float, float, float] = MISSING
    lateral_offset: float = MISSING
    top_clearance: float = MISSING
    wrist_surface_offset: float = 0.0
    dataset_path: str = MISSING
    pose_keys: tuple[str, str] = MISSING
    workspace_pose_count: int = MISSING
    orientation_weight: float = MISSING
    approach_positions: tuple = MISSING
    approach_distance: float = MISSING
    contact_distance: float = MISSING


class BoxVelocityCommand(UniformVelocityCommand):
    """Sample box velocity in the robot heading frame and measure box tracking error."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # Reset observations are computed before the next command compute() call.
        # Deliver the sampled stop immediately, including the first policy action.
        self.vel_command_b[env_ids] = torch.where(self.is_standing_env[env_ids, None], 0.0, self.vel_command_b[env_ids])

    def _update_metrics(self):
        box = self._env.scene[self.cfg.object_name]
        velocity = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w.torch), box.data.root_lin_vel_w.torch)
        self._error_xy_sum += (self.command[:, :2] - velocity[:, :2]).norm(dim=-1)
        self._error_yaw_sum += (self.command[:, 2] - box.data.root_ang_vel_w.torch[:, 2]).abs()
        self._step_count += 1


@configclass
class BoxVelocityCommandCfg(UniformVelocityCommandCfg):
    """Desired box planar velocity and world-z angular velocity."""

    class_type: type = BoxVelocityCommand
    object_name: str = MISSING
