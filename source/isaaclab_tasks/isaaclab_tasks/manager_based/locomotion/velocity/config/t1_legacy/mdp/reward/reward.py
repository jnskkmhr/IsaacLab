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

import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import RewardTermCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# class reward_symmetry(ManagerTermBase):
#     """Penalize deviation from target swing height, evaluated at landing."""

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)

#         self.cycle_torque_diff = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         asset_cfg: SceneEntityCfg,
#         std: float,
#     ) -> torch.Tensor:
        
#         asset: Articulation = env.scene[asset_cfg.name]
#         phase = env.get_phase()
#         T_not_done = torch.abs(phase - torch.round(phase)) > 0.04  # True if phase is not done
#         T_is_done = torch.abs(phase - torch.round(phase)) <= 0.04  # True if phase is done

#         error = torch.norm(self.cycle_torque_diff, dim=-1)
#         reward = torch.exp(-error / std**2) * T_is_done
#         self.cycle_torque_diff = (
#             self.cycle_torque_diff + 
#             torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids[:6]]) -
#             torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids[6:]]) * T_not_done.unsqueeze(1)
#         )
#         return reward
    
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
    original_shape = feet_quat.shape
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(feet_quat.reshape(-1, 4))

    roll = (roll + torch.pi) % (2*torch.pi) - torch.pi
    pitch = (pitch + torch.pi) % (2*torch.pi) - torch.pi
    # yaw = (yaw + torch.pi) % (2*torch.pi) - torch.pi

    return roll.reshape(original_shape[0], -1), \
                pitch.reshape(original_shape[0], -1), \
                    yaw.reshape(original_shape[0], -1)

def reward_feet_roll(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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
) -> torch.Tensor:
    # Calculate roll angles from quaternions for the feet
    _, feet_pitch, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
    )
    return torch.sum(torch.square(feet_pitch), dim=-1)

# def reward_feet_yaw(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     # Calculate roll angles from quaternions for the feet
#     _, _, feet_yaw = _feet_rpy(
#         env, 
#         asset_cfg=asset_cfg, 
#     )
#     return torch.sum(torch.square(feet_yaw), dim=-1)

"""
adapted from mjlab
"""
def feet_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity."""
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [B, N]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # [B, N, 2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    delta = torch.abs(foot_z - target_height)  # [B, N]
    cost = torch.sum(delta * vel_norm, dim=1)  # [B]
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost

# class feet_swing_height(ManagerTermBase):
#     """Penalize deviation from target swing height, evaluated at landing."""

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)

#         self.sensor_name = cfg.params["sensor_name"]
#         self.body_names = cfg.params["asset_cfg"].body_names
#         self.peak_heights = torch.zeros(
#         (env.num_envs, len(self.body_names)), device=env.device, dtype=torch.float32
#         )
#         self.step_dt = env.step_dt

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         sensor_name: str,
#         target_height: float,
#         command_name: str,
#         command_threshold: float,
#         asset_cfg: SceneEntityCfg,
#     ) -> torch.Tensor:
#         asset = env.scene[asset_cfg.name]
#         contact_sensor: ContactSensor = env.scene[sensor_name]
#         command = env.command_manager.get_command(command_name)
#         assert command is not None
#         foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
#         air_time = contact_sensor.data.current_air_time[:, asset_cfg.body_ids]
#         in_air = air_time > 0

#         self.peak_heights = torch.where(
#             in_air,
#             torch.maximum(self.peak_heights, foot_heights),
#             self.peak_heights,
#         )
#         first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)[:, asset_cfg.body_ids]
#         linear_norm = torch.norm(command[:, :2], dim=1)
#         angular_norm = torch.abs(command[:, 2])
#         total_command = linear_norm + angular_norm
#         active = (total_command > command_threshold).float()
#         error = self.peak_heights / target_height - 1.0
#         cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
#         num_landings = torch.sum(first_contact.float())
#         peak_heights_at_landing = self.peak_heights * first_contact.float()
#         mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
#         num_landings, min=1
#         )
#         env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
#         self.peak_heights = torch.where(
#             first_contact,
#             torch.zeros_like(self.peak_heights),
#             self.peak_heights,
#         )
#         return cost

def soft_landing(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    ) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.net_forces_w is not None
    forces = sensor_data.net_forces_w  # [B, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


"""
took from Feiyang's digit v3 task 
"""

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


def reward_feet_contact_number(
    env, 
    sensor_cfg: SceneEntityCfg, 
    pos_rw: float, 
    neg_rw: float
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
    stance_mask, mask_2 = create_stance_mask(phase)

    reward = torch.where(contacts == stance_mask, pos_rw, neg_rw)
    return torch.mean(reward, dim=1)

def reward_feet_contact_number_soft(
    env, 
    action_term_name: str,
    pos_rw: float, 
    neg_rw: float
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    contact_solver = env.action_manager.get_term(action_term_name).contact_solver
    contacts = contact_solver.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0] > 5.0
    # print("contact", contacts.shape, contacts)
    phase = env.get_phase()
    stance_mask, mask_2 = create_stance_mask(phase)

    reward = torch.where(contacts == stance_mask, pos_rw, neg_rw)
    return torch.mean(reward, dim=1)

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
    reward *= torch.norm(command, dim=1) > 0.05
    return reward


def track_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """"""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    command = env.command_manager.get_command(command_name)[:, :2]

    com_z = asset.data.root_pos_w[:, 2]
    standing_position_com_z = asset.data.default_root_state[:, 2]
    phase = env.get_phase()

    return compute_reward_track_foot_height(
        com_z, standing_position_com_z, phase, foot_z, 0.0305, std, command
    )


def reward_foot_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ref_dist: float
) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
    # print("foot dist", foot_dist)

    reward = torch.clip(ref_dist - foot_dist, min=0.0, max=0.1)
    
    return reward

"""
rewards feet being ariborne for 0.05-0.5 seconds
"""
def feet_air_time_positive_biped(env, command_name: str, threshold_min: float, threshold_max: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
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
    env.extras["log"]["Metrics/air_time_mean"] = air_time.mean()
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, min=threshold_min, max=threshold_max)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

"""
reimplementation of contact rewards to handle soft contact.
"""

def feet_air_time_positive_biped_soft(
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


def feet_slide_soft(
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