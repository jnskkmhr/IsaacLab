# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the `box` two-way coupling task."""

import torch


def box_target_position_reward(
    env,
    target_pos: tuple[float, float, float] = (0.0, 0.0, 0.25),
    std: float = 0.25,
) -> torch.Tensor:
    """Exponential kernel reward for stabilizing the box at the target position on the sand bed."""
    pos = env.scene["box"].data.root_pose_w.torch[:, :3] - env.scene.env_origins
    target = pos.new_tensor(target_pos).unsqueeze(0)
    dist_sq = (pos - target).square().sum(dim=-1)
    return torch.exp(-dist_sq / (std * std))


def box_height_support_reward(env, min_height: float = -0.05) -> torch.Tensor:
    """Reward keeping the box supported above the sand bed surface (matching Newton `test_final` q[2] > -0.05)."""
    z = env.scene["box"].data.root_pose_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (z > min_height).float()


def box_action_l2(env) -> torch.Tensor:
    """L2 penalty on applied wrench actions."""
    return env.action_manager.action.square().sum(dim=-1)


def box_alive_bonus(env) -> torch.Tensor:
    """Constant alive reward per step."""
    return torch.ones(env.num_envs, device=env.device)
