# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stair difficulty progression for goal policy fine-tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_reset_difficulty(
    env: ManagerBasedRLEnv,
    env_ids,
    command_name: str,
    success_step: float = 0.02,
    failure_step: float = 0.05,
) -> torch.Tensor:
    """Expand task resets only after reaching the goal and surviving the episode.

    Reference episodes and the initial reset do not update difficulty. Each
    environment progresses independently, without distributed collectives during reset.
    """
    command = env.command_manager.get_term(command_name)
    completed = (env.episode_length_buf[env_ids] > 0) & ~command.is_imitation[env_ids]
    success = command.goal_reached[env_ids] & ~env.termination_manager.terminated[env_ids]
    change = torch.where(success, success_step, -failure_step) * completed
    command.task_reset_difficulty[env_ids] = (command.task_reset_difficulty[env_ids] + change).clamp(0.0, 1.0)
    return command.task_reset_difficulty.mean()


def stair_levels(env: ManagerBasedRLEnv, env_ids, command_name: str) -> torch.Tensor:
    """Promote task goal arrivals; demote task failures, skipping imitation and initial reset."""
    command = env.command_manager.get_term(command_name)
    completed = (env.episode_length_buf[env_ids] > 0) & ~command.is_imitation[env_ids]
    success = command.goal_reached[env_ids]
    env.scene.terrain.update_env_origins(env_ids, completed & success, completed & ~success)
    return env.scene.terrain.terrain_levels.float().mean()
