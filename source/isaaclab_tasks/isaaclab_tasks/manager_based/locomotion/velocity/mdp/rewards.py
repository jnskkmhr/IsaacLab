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

import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
Foot orientation
"""

"""
feet penalties.
"""

def _feet_rpy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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

    asset = env.scene[asset_cfg.name]
    
    # Calculate roll angles from quaternions for the feet
    # feet_index = asset_cfg.body_ids
    feet_roll, _, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
        # feet_index=feet_index
    )
    
    return torch.sum(torch.square(feet_roll), dim=-1)

def reward_feet_roll_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]):
) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    
    # Calculate pitch angles from quaternions for the feet
    feet_roll, _, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
        # feet_index=feet_index
    )
    roll_rel_diff = torch.abs((feet_roll[:, 1] - feet_roll[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)
    return roll_rel_diff

def reward_feet_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]
) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    
    # Calculate roll angles from quaternions for the feet
    # feet_index = asset_cfg.body_ids
    _, feet_pitch, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
        # feet_index=feet_index
    )
    return torch.sum(torch.square(feet_pitch), dim=-1)

def reward_feet_pitch_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]):
) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    
    # Calculate pitch angles from quaternions for the feet
    _, feet_pitch, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
        # feet_index=feet_index
    )
    pitch_rel_diff = torch.abs((feet_pitch[:, 1] - feet_pitch[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)
    return pitch_rel_diff

def reward_feet_yaw_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]):
) -> torch.Tensor:
    """Reward minimizing the difference between feet yaw angles.
    
    This function rewards the agent for having similar yaw angles for all feet,
    which encourages a more stable and coordinated gait.
    
    Args:
        env: The environment.
        std: Standard deviation parameter for the exponential kernel.
        asset_cfg: Configuration for the asset.
    
    Returns:
        torch.Tensor: Reward based on similarity of feet yaw angles.
    """

    asset = env.scene[asset_cfg.name]
    
    # Calculate yaw angles from quaternions for the feet
    _, _, feet_yaw = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
        # feet_index=feet_index
    )
    yaw_rel_diff = torch.abs((feet_yaw[:, 1] - feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)
    return yaw_rel_diff

def reward_feet_yaw_mean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]
) -> torch.Tensor:

    # Get the entity
    entity = env.scene[asset_cfg.name]
    
    # Calculate yaw angles from quaternions for the feet
    _, _, feet_yaw = _feet_rpy(
        env,
        asset_cfg=asset_cfg, 
        # feet_index=feet_index
    )
    
    _, _, base_yaw = _base_rpy(
        env, asset_cfg=asset_cfg, base_index=[0]
    )
    mean_yaw = feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(feet_yaw[:, 1] - feet_yaw[:, 0]) > torch.pi)
    
    yaw_diff =  torch.abs((base_yaw - mean_yaw + torch.pi) % (2 * torch.pi) - torch.pi)
    
    return yaw_diff


"""
Gait
"""

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    if "log" in env.extras.keys():
        env.extras["log"]["Metrics/feet_air_time"] = air_time.mean()

    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

"""
Base tracking
"""

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

"""
Joint deviation
"""

def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

def body_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)