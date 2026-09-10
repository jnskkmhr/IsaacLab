# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain # type: ignore
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down) # type: ignore
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

"""
command curriculums
"""

class VelocityStage(TypedDict):
    step: int
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None
  
def commands_vel(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[VelocityStage],
    ) -> dict[str, torch.Tensor]:
    """
    Curriculum that updates the command velocity ranges based on predefined learning iterations. 
    Example: 
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
    """
    del env_ids  # Unused.
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = command_term.cfg
    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                cfg.ranges.lin_vel_x = stage["lin_vel_x"] # type: ignore
            if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                cfg.ranges.lin_vel_y = stage["lin_vel_y"] # type: ignore
            if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                cfg.ranges.ang_vel_z = stage["ang_vel_z"] # type: ignore
    return {
        "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]), # type: ignore
        "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]), # type: ignore
        "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]), # type: ignore
        "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]), # type: ignore
        "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]), # type: ignore
        "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]), # type: ignore
    }

"""
reward weight curriculums
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
        
def ramp_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    weight_0: float,
    weight_1: float,
    step_0: int,
    step_1: int,
) -> float:
    """Curriculum that continuously sets the reward weight. The weight is `weight_0` before
    `step_0`, then linearly ramps to `weight_1` at `step_1`, and is `weight_1 after `step_1`.

    This overwrites the initial weight of the term and will set the absolute value of the
    weight at every environment step so it will not play nice with other curriculum terms
    on the same reward.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight_0: The weight of the reward term below `step_0`.
        weight_1: The weight of the reward term above `step_1`.
        step_0: The step at which the weight will start to ramp.
        step_1: The step at which the weight will reach `weight_1`.
    """
    if step_0 == step_1:
        alpha = 0.0 if env.common_step_counter < step_0 else 1.0
    else:
        alpha = (env.common_step_counter - step_0) / (step_1 - step_0)
        alpha = np.clip(alpha, a_min=0.0, a_max=1.0)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = (1.0 - alpha) * weight_0 + alpha * weight_1
    env.reward_manager.set_term_cfg(term_name, term_cfg)

    return env.reward_manager.get_term_cfg(term_name).weight
        
def modify_reward_param(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, param_name: str, param_val: float, num_steps: int
):
    """Curriculum that modifies a parameter value after a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        param_name: The name of the parameter.
        param_val: The value of the parameter.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = param_val
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return env.reward_manager.get_term_cfg(term_name).params[param_name]


def ramp_reward_param(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    val_0: float,
    val_1: float,
    step_0: int,
    step_1: int,
)-> float:
    """Curriculum that continuously sets the reward parameter. The value is `val_0` before
    `step_0`, then linearly ramps to `val_1` at `step_1`, and is `val_1` after `step_1`.

    This overwrites the initial value of the term parameter and will set the absolute value of the
    parameter at every environment step so it will not play nice with other curriculum terms
    on the same reward.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        val_0: The value of the reward term below `step_0`.
        val_1: The value of the reward term above `step_1`.
        step_0: The step at which the value will start to ramp.
        step_1: The step at which the value will reach `val_1`.
    """

    if step_0 == step_1:
        alpha = 0.0 if env.common_step_counter < step_0 else 1.0
    else:
        alpha = (env.common_step_counter - step_0) / (step_1 - step_0)
        alpha = np.clip(alpha, a_min=0.0, a_max=1.0)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.params[param_name] = (1.0 - alpha) * val_0 + alpha * val_1
    env.reward_manager.set_term_cfg(term_name, term_cfg)

    return env.reward_manager.get_term_cfg(term_name).params[param_name] # type: ignore