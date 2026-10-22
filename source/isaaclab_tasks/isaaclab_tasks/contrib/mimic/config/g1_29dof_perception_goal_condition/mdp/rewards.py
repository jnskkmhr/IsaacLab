# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Reference rewards for imitation episodes and spatial rewards for task episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def imitation_reward(
    env: ManagerBasedRLEnv, command_name: str, reward_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Apply an existing motion tracking reward only to imitation episodes."""
    command = env.command_manager.get_term(command_name)
    reward_func = getattr(mimic_mdp, reward_name)
    params = {"command_name": command_name, "std": std}
    if body_names is not None:
        params["body_names"] = body_names
    return reward_func(env, **params) * command.is_imitation


def goal_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Signed change in root-to-destination distance per second in task episodes."""
    command = env.command_manager.get_term("motion")
    progress = (command.previous_goal_distance - command.goal_error.norm(dim=-1)) / env.step_dt
    return progress * ~command.is_imitation


def goal_position(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Smooth destination position reward, independent of the reference clock."""
    command = env.command_manager.get_term("motion")
    return torch.exp(-command.goal_error.square().sum(dim=-1) / std**2) * ~command.is_imitation


def goal_heading(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward the final yaw once the root is within the destination position tolerances."""
    command = env.command_manager.get_term("motion")
    return torch.exp(-command.heading_error.square() / std**2) * command.position_reached * ~command.is_imitation


def goal_settle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Discourage base motion near the destination, without penalizing the airborne jump."""
    command = env.command_manager.get_term("motion")
    velocity = command.robot.data.root_lin_vel_w.torch.square().sum(dim=-1)
    velocity += command.robot.data.root_ang_vel_w.torch.square().sum(dim=-1)
    return velocity * command.position_reached * ~command.is_imitation
