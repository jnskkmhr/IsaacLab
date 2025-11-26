# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
dof regularization penalties.
"""

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


"""
feet orientation penalties.
"""

def _feet_rpy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [0, 1]
    ):
    """Compute the yaw angles of feet.

    Args:
    env: The environment.
    asset_cfg: Configuration for the asset.
    feet_index: Optional list of indices specifying which feet to consider. 
            If None, all bodies specified in asset_cfg.body_ids are used.

    Returns:
    torch.Tensor: Yaw angles of feet in radians.
    """
    # Get the entity
    entity = env.scene[asset_cfg.name]
    
    # Get the body IDs to use
    feet_quat = entity.data.body_quat_w[:, asset_cfg.body_ids, :]
    # feet_quat = entity.data.body_quat_w[:, feet_index, :]
    original_shape = feet_quat.shape
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(feet_quat.reshape(-1, 4))

    roll = (roll + torch.pi) % (2*torch.pi) - torch.pi
    pitch = (pitch + torch.pi) % (2*torch.pi) - torch.pi
    # yaw = (yaw + torch.pi) % (2*torch.pi) - torch.pi

    return roll.reshape(original_shape[0], -1), \
                pitch.reshape(original_shape[0], -1), \
                    yaw.reshape(original_shape[0], -1)

def _base_rpy(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_index: list[int] = [0]):
    """Compute the yaw angles of feet.

    Args:
    env: The environment.
    asset_cfg: Configuration for the asset.
    feet_index: Optional list of indices specifying which feet to consider. 
            If None, all bodies specified in asset_cfg.body_ids are used.

    Returns:
    torch.Tensor: Yaw angles of feet in radians.
    """
    # Get the entity
    entity = env.scene[asset_cfg.name]
    
    # Get the body IDs to use
    body_quat = entity.data.body_quat_w[:, base_index, :]
    original_shape = body_quat.shape
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat.reshape(-1, 4))

    return roll.reshape(original_shape[0]), \
                pitch.reshape(original_shape[0]), \
                    yaw.reshape(original_shape[0])

def reward_feet_roll(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]
) -> torch.Tensor:

    # Calculate roll angles from quaternions for the feet
    feet_roll, _, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
    )
    
    return torch.sum(torch.square(feet_roll), dim=-1)

def reward_feet_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]
) -> torch.Tensor:

    # Calculate roll angles from quaternions for the feet
    _, feet_pitch, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
    )
    return torch.sum(torch.square(feet_pitch), dim=-1)

"""
gait
"""
def reward_feet_swing(    
    env: ManagerBasedRLEnv,
    swing_period: float,
    sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    freq = 1 / env.phase_dt
    phase = env.get_phase()

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        > 1.0
    )

    left_swing = (torch.abs(phase - 0.25) < 0.5 * swing_period) & (freq > 1.0e-8)
    right_swing = (torch.abs(phase - 0.75) < 0.5 * swing_period) & (freq > 1.0e-8)
    reward = (left_swing & ~contacts[:, 0]).float() + (right_swing & ~contacts[:, 1]).float()

    return reward

def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    cmd_threshold: float = 0.05,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > cmd_threshold
    return reward

def fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5

"""
contact foot penalties
"""

def reward_foot_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ref_dist: float
) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)

    reward = torch.clip(ref_dist - foot_dist, min=0.0, max=0.1)
    
    return reward

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )

def body_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward

"""
swing foot penalties
"""

def foot_clearance_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_height: float, 
    std: float, 
    tanh_mult: float, 
    standing_position_foot_z: float = 0.039,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - (target_height + standing_position_foot_z))
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


"""
reimplementation of contact rewards to handle soft contact.
"""

def feet_air_time_positive_biped(
    env, 
    command_name: str, 
    threshold: float, 
    action_term_name: str = "physics_callback",
    ) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    # compute the reward
    air_time = contact_solver.data.current_air_time # (num_envs, num_bodies)
    contact_time = contact_solver.data.current_contact_time # (num_envs, num_bodies)
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env, 
    action_term_name: str = "physics_callback",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    ) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    contacts = contact_solver.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0] > 5.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def fly_soft(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    action_term_name: str = "physics_callback",
    ) -> torch.Tensor:
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    net_contact_forces = contact_solver.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, :, :], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5

def body_force_soft(
    env: ManagerBasedRLEnv, 
    action_term_name: str = "physics_callback",
    threshold: float = 500, 
    max_reward: float = 400
) -> torch.Tensor:
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    reward = contact_solver.data.net_forces_w[:, :, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward

def reward_feet_swing_soft(    
    env: ManagerBasedRLEnv,
    swing_period: float,
    action_term_name: str = "physics_callback",
    ) -> torch.Tensor:
    freq = 1 / env.phase_dt
    phase = env.get_phase()

    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    contacts = contact_solver.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0] > 5.0

    left_swing = (torch.abs(phase - 0.25) < 0.5 * swing_period) & (freq > 1.0e-8)
    right_swing = (torch.abs(phase - 0.75) < 0.5 * swing_period) & (freq > 1.0e-8)
    reward = (left_swing & ~contacts[:, 0]).float() + (right_swing & ~contacts[:, 1]).float()

    return reward