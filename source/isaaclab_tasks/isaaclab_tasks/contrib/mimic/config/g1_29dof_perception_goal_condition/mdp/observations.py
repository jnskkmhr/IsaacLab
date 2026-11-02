# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Privileged training observations and destination symmetry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def imitation_mode(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose the reward mode to the critic, never to the actor."""
    return env.command_manager.get_term("motion").is_imitation.float().unsqueeze(-1)


def zero_reference(env: ManagerBasedRLEnv, dimension: int) -> torch.Tensor:
    """Preserve pretrained critic input slots during reference-free fine-tuning."""
    return torch.zeros(env.num_envs, dimension, device=env.device)


def reference_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reference joints for imitation returns, zeroed in task episodes."""
    command = env.command_manager.get_term("motion")
    return command.joint_pos * command.is_imitation.unsqueeze(-1)


def reference_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reference phase for imitation returns, zeroed in task episodes."""
    command = env.command_manager.get_term(command_name)
    return mimic_mdp.motion_phase(env, command_name) * command.is_imitation.unsqueeze(-1)


def motion_anchor_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reference anchor position in the robot anchor frame, zeroed in task episodes."""
    command = env.command_manager.get_term(command_name)
    return mimic_mdp.motion_anchor_pos_b(env, command_name) * command.is_imitation.unsqueeze(-1)


def motion_anchor_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reference anchor orientation in the robot anchor frame, zeroed in task episodes."""
    command = env.command_manager.get_term(command_name)
    return mimic_mdp.motion_anchor_ori_b(env, command_name) * command.is_imitation.unsqueeze(-1)


def mirror_goal(observation: torch.Tensor) -> torch.Tensor:
    """Reflect local destination Y and sine of yaw error across the robot's sagittal plane."""
    mirrored = observation.clone()
    mirrored[..., 1] *= -1
    mirrored[..., 3] *= -1
    return mirrored
