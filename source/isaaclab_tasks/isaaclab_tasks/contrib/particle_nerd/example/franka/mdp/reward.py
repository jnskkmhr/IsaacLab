# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the `franka` sand-pushing task."""

import torch


def franka_ee_sand_immersion_reward(
    env,
    target_z: float = 0.08,
    std: float = 0.10,
) -> torch.Tensor:
    """Reward keeping the `panda_hand` immersed near the sand bed surface height."""
    robot = env.scene["robot"]
    hand_idx = robot.body_names.index("panda_hand")
    z = robot.data.body_pos_w.torch[:, hand_idx, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-((z - target_z) ** 2) / (std * std))


def franka_action_l2(env) -> torch.Tensor:
    """L2 penalty on joint position actions."""
    return env.action_manager.action.square().sum(dim=-1)


def franka_alive_bonus(env) -> torch.Tensor:
    """Constant alive reward per step."""
    return torch.ones(env.num_envs, device=env.device)
