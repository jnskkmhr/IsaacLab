# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events of the tow task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def reset_object_behind_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    distance: float,
    clearance: float,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Put an object at rest behind the robot, facing the same way, just above the terrain.

    The robot must have been reset before this event runs. The object keeps the height of its default
    root pose above the terrain surface sampled at its new position.

    Args:
        env: Environment with a shared terrain surface.
        env_ids: Environment indices to reset.
        distance: Distance of the object's root behind the robot's root along the robot's heading [m].
        clearance: Extra height above the default rest height [m].
        asset_cfg: The object.
        robot_cfg: The robot.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    root_pos = robot.data.root_pos_w.torch[env_ids]
    yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w.torch[env_ids])[2]
    positions = root_pos.clone()
    positions[:, 0] -= distance * torch.cos(yaw)
    positions[:, 1] -= distance * torch.sin(yaw)
    rest_height = asset.data.default_root_pose.torch[env_ids, 2] + clearance
    if hasattr(env.scene.terrain, "background_mesh"):
        points = wp.from_torch(positions.clone().contiguous(), dtype=wp.vec3)
        env.scene.terrain.background_mesh.sample_surface_heights(points)
        positions[:, 2] = wp.to_torch(points)[:, 2] + rest_height
    else:
        positions[:, 2] = env.scene.env_origins[env_ids, 2] + rest_height
    zeros = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(positions.shape[0], 6, device=env.device), env_ids=env_ids
    )
