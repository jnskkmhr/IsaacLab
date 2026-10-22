# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Reset the robot independently of the fixed motion-paired terrain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_root_state(env: ManagerBasedRLEnv, env_ids, command_name: str) -> None:
    """Use reference initialization for imitation and perturb the task's initial root pose."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    command = env.command_manager.get_term(command_name)
    command.resample_time_steps(env_ids)
    position = command.root_pos_w[env_ids].clone()
    orientation = command.root_quat_w[env_ids].clone()
    task_mask = ~command.is_imitation[env_ids]
    position[:, :2] += torch.empty(len(env_ids), 2, device=env.device).uniform_(
        -command.cfg.initial_position_noise, command.cfg.initial_position_noise
    ) * task_mask.unsqueeze(-1)
    yaw = (
        torch.empty(len(env_ids), device=env.device).uniform_(
            -command.cfg.initial_yaw_noise, command.cfg.initial_yaw_noise
        )
        * task_mask
    )
    rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    orientation = math_utils.quat_mul(rotation, orientation)
    velocity = math_utils.quat_apply(rotation, command.root_lin_vel_w[env_ids])
    angular_velocity = math_utils.quat_apply(rotation, command.root_ang_vel_w[env_ids])
    command.robot.write_root_state_to_sim(
        torch.cat((position, orientation, velocity, angular_velocity), dim=-1), env_ids=env_ids
    )
