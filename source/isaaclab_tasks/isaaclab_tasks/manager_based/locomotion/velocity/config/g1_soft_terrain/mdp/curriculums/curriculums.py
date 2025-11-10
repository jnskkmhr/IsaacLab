# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
iteration based curriculum. 
"""

# def update_terrain_stiffness(
#     env: ManagerBasedRLEnv, 
#     env_ids: Sequence[int], 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     contact_solver_name:str="physics_callback",
# ):
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
#     terrain: TerrainImporter = env.scene.terrain
#     command = env.command_manager.get_command("base_velocity")
    
#     # compute the distance the robot walked
#     distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
#     # robots that walked far enough progress to harder terrains
#     move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
#     # robots that walked less than half of their required distance go to simpler terrains
#     move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
#     move_down *= ~move_up
    
#     contact_solver.update_ground_stiffness(env_ids, move_up, move_down)
#     print("stiffness: ", torch.mean(contact_solver.stiffness.float()))
#     return torch.mean(contact_solver.stiffness.float())


def update_terrain_stiffness(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_solver_name:str="physics_callback",
):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    scale = 0.3
    move_up = distance > terrain.cfg.terrain_generator.size[0] * scale
    # robots that walked less than predefined distance threshold go to more rigid terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    
    contact_solver.update_ground_stiffness(env_ids, move_up, move_down)
    return torch.mean(contact_solver.stiffness.float())

def terrain_ground_level(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    rl_horizon: int = 24,
    max_iterations: int = 1500,
    minimum_ground_height: float = -0.1, 
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain
    
    # compute how many learning iterations have passed
    env_step_count = env._sim_step_counter // env.cfg.decimation
    iteration = env_step_count // rl_horizon
    progress = (iteration / max_iterations)**4
    progress = min(progress, 1.0)
    ground_height = progress * minimum_ground_height

    # update ground height
    terrain.update_terrain_height(ground_height)

    # return the mean terrain level
    return torch.tensor([ground_height])