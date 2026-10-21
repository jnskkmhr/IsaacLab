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
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat
from isaaclab.utils.string import resolve_matching_names_values

from isaaclab_tasks.contrib.velocity.config.vel_mdp.rewards import capped_foot_clearance_reward
from isaaclab_tasks.contrib.velocity.config.vel_mdp.rewards import feet_edge_penalty as feet_edge_penalty

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import TerrainPoseCommand


class goal_progress(ManagerTermBase):
    """Reward decreasing horizontal base-to-goal distance, measured in meters per second."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_distance = torch.zeros(env.num_envs, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, max_speed: float = 1.5) -> torch.Tensor:
        command: TerrainPoseCommand = env.command_manager.get_term(command_name)
        distance = torch.linalg.vector_norm(command.position_error_w[:, :2], dim=-1)
        progress = ((self.previous_distance - distance) / env.step_dt).clamp(-max_speed, max_speed)
        progress = torch.where(self.valid, progress, 0.0)
        self.previous_distance[:] = distance
        self.valid[:] = True
        return progress


class base_acceleration_l2(ManagerTermBase):
    """Penalize horizontal linear or yaw acceleration over a control step.

    Difference world-frame root velocities so changing the robot's heading does
    not introduce artificial acceleration. Each reward term owns its history;
    the first evaluation after an environment reset contributes zero.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_velocity = torch.zeros(env.num_envs, 3, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, angular: bool = False) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        velocity = asset.data.root_ang_vel_w.torch if angular else asset.data.root_lin_vel_w.torch
        acceleration = (velocity - self.previous_velocity) / env.step_dt
        acceleration = acceleration[:, 2:] if angular else acceleration[:, :2]
        penalty = torch.where(self.valid, acceleration.square().sum(dim=-1), 0.0)
        self.previous_velocity[:] = velocity
        self.valid[:] = True
        return penalty


def swing_foot_lateral_position_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    lateral_tolerance: float,
) -> torch.Tensor:
    """Penalize swing feet outside a lateral band around their corresponding hips.

    Measure ankle-roll body positions relative to hip-pitch body origins in the
    robot's yaw-aligned frame. Forward and vertical displacement are unpenalized.
    ``lateral_tolerance`` is the allowed absolute lateral displacement in meters.
    Feet, hips, and contact bodies must all be selected in left/right order.
    """
    asset = env.scene[asset_cfg.name]
    relative_position = (
        asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - asset.data.body_pos_w.torch[:, hip_cfg.body_ids]
    )
    heading = yaw_quat(asset.data.root_quat_w.torch).unsqueeze(1).expand(-1, relative_position.shape[1], -1)
    lateral_position = quat_apply_inverse(heading, relative_position)[..., 1]
    excess = (lateral_position.abs() - lateral_tolerance).clamp(min=0.0)
    sensor = env.scene.sensors[sensor_cfg.name]
    swing = sensor.data.current_air_time.torch[:, sensor_cfg.body_ids] > 0.0
    return (excess.square() * swing).sum(dim=-1)


class variable_posture_l1(ManagerTermBase):
    """Penalize joint deviation using separate weights for approaching and holding the goal."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        self.default_joint_pos = asset.data.default_joint_pos.torch
        _, joint_names = asset.find_joints(asset_cfg.joint_names, preserve_order=asset_cfg.preserve_order)
        _, _, weight_hold = resolve_matching_names_values(cfg.params["weight_hold"], joint_names)
        self.weight_hold = torch.tensor(weight_hold, device=env.device, dtype=torch.float32)
        _, _, weight_moving = resolve_matching_names_values(cfg.params["weight_moving"], joint_names)
        self.weight_moving = torch.tensor(weight_moving, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        weight_hold: dict[str, float],
        weight_moving: dict[str, float],
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        command: TerrainPoseCommand = env.command_manager.get_term(command_name)
        command.update_goal_status()
        weight = torch.where(command.goal_in_tolerance.unsqueeze(-1), self.weight_hold, self.weight_moving)
        error = torch.abs(
            asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - self.default_joint_pos[:, asset_cfg.joint_ids]
        )
        return (weight * error).sum(dim=-1)


def goal_position_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward 3D base-position accuracy, including target Z."""
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    return torch.exp(-command.position_error_w.square().sum(dim=-1) / std**2)


def goal_heading_exp(env: ManagerBasedRLEnv, command_name: str, std: float, position_std: float) -> torch.Tensor:
    """Reward target yaw, emphasizing orientation once the base is near the goal."""
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    return goal_position_exp(env, command_name, position_std) * torch.exp(-command.heading_error.square() / std**2)


def goal_facing_exp(env: ManagerBasedRLEnv, command_name: str, std: float, distance_threshold: float) -> torch.Tensor:
    """Reward facing the horizontal direction toward the goal during approach.

    The weight fades linearly from one at ``distance_threshold`` to zero at the
    command's XY position tolerance, leaving final goal yaw unconstrained by this term.
    """
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    error_xy = command.position_error_w[:, :2]
    heading = torch.atan2(error_xy[:, 1], error_xy[:, 0])
    heading_error = wrap_to_pi(heading - command.robot.data.heading_w.torch)
    distance = torch.linalg.vector_norm(error_xy, dim=-1)
    weight = (
        (distance - command.cfg.position_tolerance) / (distance_threshold - command.cfg.position_tolerance)
    ).clamp(0.0, 1.0)
    return weight * torch.exp(-heading_error.square() / std**2)


def goal_motion_l2(env: ManagerBasedRLEnv, command_name: str, position_std: float) -> torch.Tensor:
    """Penalize base and foot motion near the target base position."""
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    motion = command.robot.data.root_lin_vel_w.torch.square().sum(dim=-1)
    motion += command.robot.data.root_ang_vel_w.torch.square().sum(dim=-1)
    motion += command.robot.data.body_lin_vel_w.torch[:, command.foot_ids].square().sum(dim=(-1, -2))
    return goal_position_exp(env, command_name, position_std) * motion


def goal_hold(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward being within the goal pose tolerances, including the final step before reset."""
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    command.update_goal_status()
    return command.goal_in_tolerance.float()


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


def fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize both feet losing contact, using contact history as in the velocity task.

    Contact requires force magnitude above ``threshold`` newtons in any history
    sample. Apply at every goal distance, without a velocity-command gate.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    in_contact = torch.linalg.vector_norm(forces, dim=-1).amax(dim=1) > threshold
    return ~in_contact.any(dim=-1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    reference_foot_height: float,
    sole_offset: tuple[float, float, float],
    tanh_mult: float,
) -> torch.Tensor:
    """Reward moving swing feet for terrain-relative clearance, capped at the reference height.

    Clearance is the sole-center world Z minus the nearest valid terrain ray hit's
    world Z. ``sole_offset`` locates that center in the ankle-roll body's frame.
    Robot and contact selections must contain left and right feet in that order.
    Only single-support swings outside the goal's XY/yaw tolerances earn reward;
    horizontal foot speed weights the reward, and missing terrain hits earn zero.
    """
    reward = capped_foot_clearance_reward(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        left_sensor_cfg=left_sensor_cfg,
        right_sensor_cfg=right_sensor_cfg,
        reference_foot_height=reference_foot_height,
        sole_offset=sole_offset,
        tanh_mult=tanh_mult,
    )
    command: TerrainPoseCommand = env.command_manager.get_term(command_name)
    needs_step = (
        torch.linalg.vector_norm(command.position_error_w[:, :2], dim=-1) >= command.cfg.position_tolerance
    ) | (command.heading_error.abs() >= command.cfg.heading_tolerance)
    return reward * needs_step
