# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Reset the robot independently of the fixed motion-paired terrain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.utils import math as math_utils

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp

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
    if command.cfg.task_start_offset is not None:
        task_ids = env_ids[task_mask]
        reference_yaw = math_utils.yaw_quat(command.root_quat_w[task_ids])
        offset = torch.tensor(command.cfg.task_start_offset, device=env.device).expand(len(task_ids), -1).clone()
        blend = torch.ones(len(task_ids), device=env.device)
        scale = torch.full_like(blend, command.reset_randomization_scale)
        if command.cfg.task_reset_curriculum:
            blend = command.task_reset_difficulty[task_ids]
            scale = blend
        offset *= blend[:, None]
        offset[:, :2] += (
            torch.empty(len(task_ids), 2, device=env.device).uniform_(
                -command.cfg.initial_position_noise, command.cfg.initial_position_noise
            )
            * scale[:, None]
        )
        position[task_mask] = command.root_pos_w[task_ids] + math_utils.quat_apply(reference_yaw, offset)
        yaw = (
            torch.empty(len(task_ids), device=env.device).uniform_(
                -command.cfg.initial_yaw_noise, command.cfg.initial_yaw_noise
            )
            * scale
        )
        roll_ref, pitch_ref, yaw_ref = math_utils.euler_xyz_from_quat(command.root_quat_w[task_ids])
        orientation[task_mask] = math_utils.quat_from_euler_xyz(
            roll_ref * (1.0 - blend), pitch_ref * (1.0 - blend), yaw_ref + yaw
        )
        velocity_scale = min(
            max((env.common_step_counter - command.cfg.task_velocity_start_step) / command.cfg.transition_steps, 0.0),
            1.0,
        )
        ranges = torch.tensor(
            [command.cfg.task_velocity_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=env.device,
        )
        samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(task_ids), 6), env.device)
        moving = torch.rand(len(task_ids), device=env.device) < command.cfg.task_velocity_probability
        if command.cfg.task_reset_curriculum:
            # Perturb velocities only after most reference momentum has been removed.
            velocity_scale = ((blend - 0.8) / 0.2).clamp(0.0, 1.0)[:, None]
        samples *= velocity_scale * moving[:, None]
        velocity[task_mask] = command.root_lin_vel_w[task_ids] * (1.0 - blend[:, None])
        velocity[task_mask] += math_utils.quat_apply(reference_yaw, samples[:, :3])
        angular_velocity[task_mask] = command.root_ang_vel_w[task_ids] * (1.0 - blend[:, None])
        angular_velocity[task_mask] += math_utils.quat_apply(reference_yaw, samples[:, 3:])
    command.robot.write_root_state_to_sim(
        torch.cat((position, orientation, velocity, angular_velocity), dim=-1), env_ids=env_ids
    )


def reset_joint_state(
    env: ManagerBasedRLEnv, env_ids, command_name: str, position_range: tuple[float, float] = (-0.1, 0.1)
) -> None:
    """Keep imitation reference resets and optionally start task episodes standing at rest."""
    command = env.command_manager.get_term(command_name)
    if command.cfg.task_start_offset is None:
        mimic_mdp.reset_joint_state_from_reference(env, env_ids, command_name, position_range)
    elif command.cfg.task_reset_curriculum:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        command.resample_time_steps(env_ids)
        blend = (command.task_reset_difficulty[env_ids] * ~command.is_imitation[env_ids])[:, None]
        robot = command.robot
        joint_pos = torch.lerp(command.joint_pos_robot_order[env_ids], robot.data.default_joint_pos[env_ids], blend)
        joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, env.device)
        limits = robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, limits[:, :, 0], limits[:, :, 1])
        joint_vel = command.joint_vel_robot_order[env_ids] * (1.0 - blend)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    else:
        reset_mixed_stair_joints(env, env_ids, command_name, position_range)


def reset_stair_root_state(env: ManagerBasedRLEnv, env_ids, command_name: str) -> None:
    """Start at rest on a flat ground border, facing up the staircase."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    command = env.command_manager.get_term(command_name)
    position, orientation = command.sample_spawn(env_ids)
    velocity = torch.zeros(len(env_ids), 6, device=env.device)
    command.robot.write_root_state_to_sim(torch.cat((position, orientation, velocity), dim=-1), env_ids=env_ids)


def reset_mixed_stair_joints(
    env: ManagerBasedRLEnv, env_ids, command_name: str, position_range: tuple[float, float] = (-0.1, 0.1)
) -> None:
    """Use reference joints for imitation and a standing pose for randomized stair tasks."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    command = env.command_manager.get_term(command_name)
    command.resample_time_steps(env_ids)
    imitation_ids = env_ids[command.is_imitation[env_ids]]
    task_ids = env_ids[~command.is_imitation[env_ids]]
    mimic_mdp.reset_joint_state_from_reference(env, imitation_ids, command_name, position_range)
    mdp.reset_joints_by_offset(env, task_ids, position_range=position_range, velocity_range=(0.0, 0.0))


def reset_mixed_stair_root_state(env: ManagerBasedRLEnv, env_ids, command_name: str) -> None:
    """Match the root reset to the sampled episode type and terrain."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    command = env.command_manager.get_term(command_name)
    command.resample_time_steps(env_ids)
    reset_root_state(env, env_ids[command.is_imitation[env_ids]], command_name)
    reset_stair_root_state(env, env_ids[~command.is_imitation[env_ids]], command_name)
