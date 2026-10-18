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
from typing import TYPE_CHECKING

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
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_motion_success(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum based on motion reference completion before termination.

    This term is used to increase the difficulty of the terrain when the robot successfully completes
    the motion reference before any termination condition is triggered. Incomplete episodes or simultaneous
    failures move the robot to an easier terrain level.

    Args:
        env: The environment.
        env_ids: The environment IDs to update.
        command_name: The name of the motion command to track. Defaults to "motion".
        asset_cfg: The scene entity configuration for the robot asset. Defaults to "robot".

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    terrain: TerrainImporter = env.scene.terrain
    motion_command = env.command_manager.get_term(command_name)

    # Check if motion was completed before termination. The curriculum manager runs at the top of
    # `_reset_idx`, before the command term resamples, so this still sees the episode's final frame.
    motion_completed = motion_command.has_reached_end[env_ids]

    # Move up to harder terrain if motion was successfully completed
    move_up = motion_completed & ~env.termination_manager.terminated[env_ids]
    move_down = ~move_up

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def assistive_wrench_scale(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str = "assistive_wrench",
    start_step: int = 0,
    end_step: int = 10000,
    initial_scale: float = 1.0,
    final_scale: float = 0.0,
) -> float:
    """Linearly anneal the assistive wrench's ``scale`` (beta) with the environment step count.

    The wrench carries the robot through the motion early on, when the policy cannot yet do it alone,
    and is faded out so the final policy is not leaning on a force it will not have at deployment.

    This is the open-loop fallback: it anneals on a wall-clock schedule with no idea whether the policy
    has actually learned anything yet, and its step counts are in *environment* steps, so a schedule
    written in RSL-RL iterations must be multiplied by ``num_steps_per_env``. Prefer the ZEST S6
    automatic curriculum -- ``AssistiveWrench(adaptive_scale=True)``, driven by
    :attr:`MotionCommand.bin_assist_scale` -- which ties the gain to the measured per-bin failure rate.
    The two compose: ``scale`` multiplies the adaptive gain when both are enabled.

    Args:
        env: The environment.
        env_ids: The environment IDs being reset (unused, the scale is global).
        term_name: The name of the ``AssistiveWrench`` event term to update.
        start_step: Environment step at which the anneal begins.
        end_step: Environment step at which the scale reaches ``final_scale``.
        initial_scale: Beta before ``start_step``.
        final_scale: Beta after ``end_step``.

    Returns:
        The current assist scale, for logging.
    """
    alpha = (env.common_step_counter - start_step) / max(end_step - start_step, 1)
    alpha = min(max(alpha, 0.0), 1.0)
    scale = initial_scale + alpha * (final_scale - initial_scale)

    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["scale"] = scale
    env.event_manager.set_term_cfg(term_name, term_cfg)
    return scale
