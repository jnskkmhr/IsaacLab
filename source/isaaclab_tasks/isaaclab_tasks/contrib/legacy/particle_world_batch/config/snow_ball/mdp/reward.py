# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the `snow_ball` slope rolling task (`example_mpm_snow_ball.py`)."""

import torch


def snow_ball_downslope_velocity_reward(
    env,
    target_speed: float = 0.35,
    std: float = 0.25,
) -> torch.Tensor:
    """Reward rolling smoothly down the +x snow slope at `target_speed`."""
    vx = env.scene["snow_ball"].data.root_vel_w.torch[:, 0]
    return torch.exp(-((vx - target_speed) ** 2) / (std * std))


def snow_ball_track_centerline_reward(env, std: float = 0.20) -> torch.Tensor:
    """Reward keeping the rolling snow-ball aligned with the slope centerline (y = 0)."""
    y = env.scene["snow_ball"].data.root_pose_w.torch[:, 1] - env.scene.env_origins[:, 1]
    return torch.exp(-(y * y) / (std * std))


def snow_ball_action_l2(env) -> torch.Tensor:
    """L2 penalty on applied external wrench actions."""
    return env.action_manager.action.square().sum(dim=-1)


def snow_ball_alive_bonus(env) -> torch.Tensor:
    """Constant alive reward per step."""
    return torch.ones(env.num_envs, device=env.device)
