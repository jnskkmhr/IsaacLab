# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the `sphere` two-way coupling task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sphere_target_position_reward(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float] = (0.0, 0.0, 0.25),
    std: float = 0.25,
) -> torch.Tensor:
    """Exponential kernel reward for stabilizing the sphere at the target position on the sand bed."""
    pos = env.scene["sphere"].data.root_pose_w.torch[:, :3] - env.scene.env_origins
    target = pos.new_tensor(target_pos).unsqueeze(0)
    dist_sq = (pos - target).square().sum(dim=-1)
    return torch.exp(-dist_sq / (std * std))


def sphere_height_support_reward(env: ManagerBasedRLEnv, min_height: float = -0.05) -> torch.Tensor:
    """Reward keeping the sphere supported above the sand bed surface."""
    z = env.scene["sphere"].data.root_pose_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (z > min_height).float()


def sphere_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on applied wrench actions."""
    return env.action_manager.action.square().sum(dim=-1)


def sphere_alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant alive reward per step."""
    return torch.ones(env.num_envs, device=env.device)
