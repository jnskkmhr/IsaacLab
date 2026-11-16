# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the `go2` quadruped granular locomotion task."""

import torch


def go2_forward_vel_reward(env, target_vx: float = 0.20, std: float = 0.25) -> torch.Tensor:
    """Exponential kernel reward for tracking forward velocity along +X over sand."""
    vx = env.scene["robot"].data.root_lin_vel_b.torch[:, 0]
    return torch.exp(-((vx - target_vx) ** 2) / (std * std))


def go2_base_height_reward(env, target_z: float = 0.34, std: float = 0.08) -> torch.Tensor:
    """Reward maintaining nominal quadruped standing height on the sand bed."""
    z = env.scene["robot"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-((z - target_z) ** 2) / (std * std))


def go2_action_l2(env) -> torch.Tensor:
    """L2 penalty on joint position actions."""
    return env.action_manager.action.square().sum(dim=-1)


def go2_alive_bonus(env) -> torch.Tensor:
    """Constant alive reward per step."""
    return torch.ones(env.num_envs, device=env.device)
