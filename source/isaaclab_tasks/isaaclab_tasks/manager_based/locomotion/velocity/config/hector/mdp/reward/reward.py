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
from typing import TYPE_CHECKING, Tuple

from isaaclab.envs import mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def track_torso_height_exp(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("sensor"), 
    reference_height: float=0.5, 
    std:float=0.5) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    contacts = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=2) > 1.0).float()
    root_pos_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    body_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = (root_pos_z - contacts*body_pos_z).max(dim=1).values
    
    reward = torch.exp(-torch.square(height - reference_height)/std**2) # exponential reward
    return reward


def torso_height_l2(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("sensor"), 
    reference_height: float=0.5) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.
    """
    
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    contacts = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=2) > 1.0).float()
    root_pos_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    body_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = (root_pos_z - contacts*body_pos_z).max(dim=1).values
    
    return torch.square(height - reference_height)


def torso_height_l1(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("sensor"), 
    reference_height: float=0.5) -> torch.Tensor:
    """Penalize asset height from its target using L1 squared kernel.
    """
    
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    contacts = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=2) > 1.0).float()
    root_pos_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    body_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = (root_pos_z - contacts*body_pos_z).max(dim=1).values
    return torch.abs(height - reference_height)


@torch.jit.script
def create_stance_mask(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a stance mask based on the gait phase.
    """
    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(-1).repeat(1, 2)
    stance_mask = torch.where(sin_pos >= 0, 1, 0)
    stance_mask[:, 1] = 1 - stance_mask[:, 1]
    stance_mask[torch.abs(sin_pos) < 0.1] = 1

    mask_2 = 1 - stance_mask
    mask_2[torch.abs(sin_pos) < 0.1] = 1
    return stance_mask, mask_2


@torch.jit.script
def compute_reward_reward_feet_contact_number(
    contacts: torch.Tensor,
    phase: torch.Tensor,
    pos_rw: float,
    neg_rw: float,
    command: torch.Tensor,
):

    stance_mask, mask_2 = create_stance_mask(phase)

    reward = torch.where(contacts == stance_mask, pos_rw, neg_rw)
    reward = torch.mean(reward, dim=1)
    # no reward for zero command
    reward *= torch.norm(command, dim=1) > 0.1
    return reward


def reward_feet_contact_number(
    env,
    sensor_cfg: SceneEntityCfg,
    pos_rw: float,
    neg_rw: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    # print("contact", contacts.shape, contacts)
    phase = env.get_phase()
    command = env.command_manager.get_command(command_name)[:, :2]

    return compute_reward_reward_feet_contact_number(
        contacts, phase, pos_rw, neg_rw, command
    )


@torch.jit.script
def compute_reward_foot_clearance_reward(
    com_z: torch.Tensor,
    standing_position_com_z: torch.Tensor,
    current_foot_z: torch.Tensor,
    target_height: float,
    std: float,
    tanh_mult: float,
    body_lin_vel_w: torch.Tensor,
    command: torch.Tensor,
):
    standing_height = com_z - standing_position_com_z
    standing_position_toe_roll_z = (
        0.0626  # recorded from the default position, 0.1 compensation for walking
    )
    offset = (standing_height + standing_position_toe_roll_z).unsqueeze(-1)
    foot_z_target_error = torch.square(
        (current_foot_z - (target_height + offset).repeat(1, 2)).clip(max=0.0)
    )
    # weighted by the velocity of the feet in the xy plane
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(body_lin_vel_w, dim=2))
    reward = foot_velocity_tanh * foot_z_target_error
    reward = torch.exp(-torch.sum(reward, dim=1) / std)
    reward *= torch.norm(command, dim=1) > 0.1
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a specified height off the ground
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w[:, 2]
    current_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    standing_position_com_z = asset.data.default_root_state[:, 2]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    command = env.command_manager.get_command(command_name)[:, :2]

    return compute_reward_foot_clearance_reward(
        com_z,
        standing_position_com_z,
        current_foot_z,
        target_height,
        std,
        tanh_mult,
        body_lin_vel_w,
        command,
    )


@torch.jit.script
def height_target(t: torch.Tensor):
    a5, a4, a3, a2, a1, a0 = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]
    return a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0


@torch.jit.script
def compute_reward_track_foot_height(
    com_z: torch.Tensor,
    standing_position_com_z: torch.Tensor,
    phase: torch.Tensor,
    foot_z: torch.Tensor,
    standing_position_toe_roll_z: float,
    std: float,
    command: torch.Tensor,
):

    standing_height = com_z - standing_position_com_z

    offset = standing_height + standing_position_toe_roll_z

    stance_mask, mask_2 = create_stance_mask(phase)

    swing_mask = 1 - stance_mask

    filt_foot = torch.where(swing_mask == 1, foot_z, torch.zeros_like(foot_z))

    phase_mod = torch.fmod(phase, 0.5)
    feet_z_target = height_target(phase_mod) + offset
    feet_z_value = torch.sum(filt_foot, dim=1)

    error = torch.square(feet_z_value - feet_z_target)
    reward = torch.exp(-error / std**2)
    # no reward for zero command
    reward *= torch.norm(command, dim=1) > 0.1
    return reward


def track_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """"""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    command = env.command_manager.get_command(command_name)[:, :2]
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    # contacts = (
    #     contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
    #     .norm(dim=-1)
    #     .max(dim=1)[0]
    #     > 1.0
    # )
    com_z = asset.data.root_pos_w[:, 2]
    standing_position_com_z = asset.data.default_root_state[:, 2]
    phase = env.get_phase()

    return compute_reward_track_foot_height(
        com_z, standing_position_com_z, phase, foot_z, 0.0486, std, command
    )