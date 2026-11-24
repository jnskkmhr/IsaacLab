# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events of the soft-contact tow task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def reset_object_behind_robot_on_ground(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distance: float,
    clearance: float,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Put an object behind the robot along its heading, at rest height above world z=0.

    The robot must have been reset before this event runs. Rigid ground and the analytical sand surface are both
    at world z=0; the rigid support under the sand lies lower and is not the surface the object rests on.

    Args:
        env: Environment instance.
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
    positions[:, 2] = asset.data.default_root_pose.torch[env_ids, 2] + clearance
    zeros = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(positions.shape[0], 6, device=env.device), env_ids=env_ids
    )
