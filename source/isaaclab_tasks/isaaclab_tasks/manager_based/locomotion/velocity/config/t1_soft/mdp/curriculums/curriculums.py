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
from typing import TYPE_CHECKING, ClassVar

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
command curriculums.
"""

def modify_reward_std(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, std: float, num_steps: int):
    """Curriculum that modifies a exponential reward std a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        std: The std of the exponential reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params["std"] = std
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_command_range(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, ranges: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.command_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.ranges = ranges
        env.command_manager.set_term_cfg(term_name, term_cfg)

"""
terrain curriculums.
"""

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
    # print("stiffness: ", torch.mean(contact_solver.stiffness.float()))
    return torch.mean(contact_solver.stiffness.float())