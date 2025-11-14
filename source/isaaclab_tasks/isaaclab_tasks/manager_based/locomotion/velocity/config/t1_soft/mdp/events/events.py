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

"""
dof reset
"""

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
    joint_pos[:, joint_indices] = target_pos
    joint_vel[:, joint_indices] = target_vel

    # Set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)

"""
terrain physical parameters randomization
"""

def randomize_terrain_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
    contact_solver_name:str="physics_callback",
)->None:
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    friction_samples = math_utils.sample_uniform(
        friction_range[0], friction_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.update_friction_params(env_ids, friction_samples, friction_samples)

def randomize_terrain_stiffness(
    env: ManagerBasedEnv, 
    env_ids: torch.Tensor, 
    stiffness_range: tuple[float, float],
    contact_solver_name:str="physics_callback",
)->None:
    # extract the used quantities (to enable type-hinting)
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    stiffness_samples = math_utils.sample_uniform(
        stiffness_range[0], stiffness_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.randomize_ground_stiffness(env_ids, stiffness_samples)