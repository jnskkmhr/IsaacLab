# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose-reaching objectives, independent of velocity commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import TerrainPoseCommand


class goal_progress(ManagerTermBase):
    """Reward decreasing 3D base-to-goal distance, measured in meters per second."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_distance = torch.zeros(env.num_envs, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, max_speed: float = 1.5) -> torch.Tensor:
        command: TerrainPoseCommand = env.command_manager.get_term(command_name)
        distance = torch.linalg.vector_norm(command.position_error_w, dim=-1)
        progress = ((self.previous_distance - distance) / env.step_dt).clamp(-max_speed, max_speed)
        progress = torch.where(self.valid, progress, 0.0)
        self.previous_distance[:] = distance
        self.valid[:] = True
        return progress


def goal_position_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward 3D base-position accuracy, including target Z."""
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    return torch.exp(-command.position_error_w.square().sum(dim=-1) / std**2)


def goal_heading_exp(env: ManagerBasedRLEnv, command_name: str, std: float, position_std: float) -> torch.Tensor:
    """Reward target yaw, emphasizing orientation once the base is near the goal."""
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    return goal_position_exp(env, command_name, position_std) * torch.exp(-command.heading_error.square() / std**2)


def goal_motion_l2(env: ManagerBasedRLEnv, command_name: str, position_std: float) -> torch.Tensor:
    """Penalize base and foot motion near the target base position."""
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    motion = command.robot.data.root_lin_vel_w.torch.square().sum(dim=-1)
    motion += command.robot.data.root_ang_vel_w.torch.square().sum(dim=-1)
    motion += command.robot.data.body_lin_vel_w.torch[:, command.foot_ids].square().sum(dim=(-1, -2))
    return goal_position_exp(env, command_name, position_std) * motion


def goal_hold(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward holding the goal; update success even on the final step before reset."""
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    command.update_goal_status()
    return command.goal_held.float()


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward single-foot stance while approaching or turning toward the goal.

    Use the command's XY and yaw tolerances to stop rewarding steps once the
    robot can settle. Height error alone does not require lifting a foot.
    As in the velocity task, the reward is the shorter of the stance and swing
    durations, capped at ``threshold`` seconds; double support and flight get zero.
    """
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    sensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = in_contact.sum(dim=-1) == 1
    reward = torch.where(single_stance, in_mode_time.min(dim=-1).values, 0.0).clamp(max=threshold)
    needs_step = (
        torch.linalg.vector_norm(command.position_error_w[:, :2], dim=-1) >= command.cfg.position_tolerance
    ) | (command.heading_error.abs() >= command.cfg.heading_tolerance)
    return reward * needs_step
