# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
import random
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypedDict

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

def randomize_terrain_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
    contact_solver_name: str = "physics_callback",
) -> None:
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver # type: ignore
    friction_samples = math_utils.sample_uniform(
        friction_range[0], friction_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.update_friction_params(env_ids, friction_samples, friction_samples)
    if "log" in env.extras.keys():
        if "Events/terrain_friction" in env.extras["log"].keys():
            env.extras["log"]["Events/terrain_friction"] = friction_samples.mean()


def randomize_terrain_stiffness(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    stiffness_range: tuple[float, float],
    contact_solver_name: str = "physics_callback",
) -> None:
    # extract the used quantities (to enable type-hinting)
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver # type: ignore
    stiffness_samples = math_utils.sample_uniform(
        stiffness_range[0], stiffness_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.randomize_ground_stiffness(env_ids, stiffness_samples)
    if "log" in env.extras.keys():
        if "Events/terrain_stiffness" in env.extras["log"].keys():
            env.extras["log"]["Events/terrain_stiffness"] = stiffness_samples.mean()
            
def randomize_cone_model_terrain_stiffness(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    sigma_flat_range: tuple[float, float],
    sigma_cone_range: tuple[float, float],
    contact_solver_name: str = "physics_callback",
) -> None:
    # extract the used quantities (to enable type-hinting)
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver # type: ignore
    sigma_flat_samples = math_utils.sample_uniform(
        sigma_flat_range[0], sigma_flat_range[1], (len(env_ids),), device=env.device
    )
    sigma_cone_samples = math_utils.sample_uniform(
        sigma_cone_range[0], sigma_cone_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.randomize_ground_stiffness(env_ids, sigma_flat_samples, sigma_cone_samples)
    if "log" in env.extras.keys():
        if "Events/terrain_stiffness" in env.extras["log"].keys():
            env.extras["log"]["Events/sigma_flat"] = sigma_flat_samples.mean()
        if "Events/sigma_cone" in env.extras["log"].keys():
            env.extras["log"]["Events/sigma_cone"] = sigma_cone_samples.mean()


def randomize_material_density(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    packing_ratio_range: tuple[float, float],
    bulk_density_range: tuple[float, float],
    contact_solver_name: str = "physics_callback",
) -> None:
    # extract the used quantities (to enable type-hinting)
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver # type: ignore
    packing_ratio = math_utils.sample_uniform(
        packing_ratio_range[0], packing_ratio_range[1], (len(env_ids),), device=env.device
    )
    bulk_density = math_utils.sample_uniform(
        bulk_density_range[0], bulk_density_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.update_material_density(env_ids, packing_ratio, bulk_density)
    if "log" in env.extras.keys():
        if "Events/packing_ratio" in env.extras["log"].keys():
            env.extras["log"]["Events/packing_ratio"] = packing_ratio.mean()
        if "Events/bulk_density" in env.extras["log"].keys():
            env.extras["log"]["Events/bulk_density"] = bulk_density.mean()


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # tile instead of random sample
    pos_x = torch.linspace(range_list[0][0], range_list[0][1], steps=math.ceil(len(env_ids)**(1/2)), device=asset.device)
    pos_y = torch.linspace(range_list[1][0], range_list[1][1], steps=math.ceil(len(env_ids)**(1/2)), device=asset.device)
    grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing='ij')
    rand_samples[0:len(grid_x.flatten()), 0] = grid_x.flatten()[0:len(env_ids)]
    rand_samples[0:len(grid_y.flatten()), 1] = grid_y.flatten()[0:len(env_ids)]

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids) # type: ignore
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids) # type: ignore