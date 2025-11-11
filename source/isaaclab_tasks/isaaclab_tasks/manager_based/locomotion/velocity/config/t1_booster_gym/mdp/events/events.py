# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

def reset_robot_upper_joints_from_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # envCfg:ManagerBasedEnvCfg,
    # joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset the designated upper body joints to 90% of their joint limits.    
    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        joint_names: List of joint names to reset.
        asset_cfg: SceneEntityCfg for the robot asset.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get joint indices for the specified joint names
    # joint_indices = []
    # for name in joint_names:
    #     idx = asset.joint_names.index(name) if name in asset.joint_names else None
    #     if idx is not None:
    #         joint_indices.append(idx)
    # if not joint_indices:
    #     return  # nothing to do    # joint_indices = torch.tensor(joint_indices, device=asset.device, dtype=torch.long)
    joint_indices = asset_cfg.joint_ids    # Get joint limits for the selected joints
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_indices, :]
    # Compute 90% of the way from lower to upper limit
    lower = joint_pos_limits[..., 0]
    upper = joint_pos_limits[..., 1]
    # Randomly sample from lower + 0.1*(upper-lower) to lower + 0.9*(upper-lower) for each joint
    min_pos = lower + 0.1 * (upper - lower)
    max_pos = lower + 0.9 * (upper - lower)
    target_pos = math_utils.sample_uniform(min_pos, max_pos, lower.shape, device=lower.device)
    # Set velocities to zero for these joints
    target_vel = torch.zeros_like(target_pos, device=asset.device)

    # Get current joint positions/velocities for all joints
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()    
    # Overwrite only the selected joints
    joint_pos[:, joint_indices] += target_pos
    joint_vel[:, joint_indices] = target_vel

    # Set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)