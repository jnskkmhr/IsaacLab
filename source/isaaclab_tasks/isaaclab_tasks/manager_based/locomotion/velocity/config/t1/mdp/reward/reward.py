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

"""
rewards to enforce 
- symmetric contact sequence
- polynomial foot trajectory
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
    env, sensor_cfg: SceneEntityCfg, pos_rw: float, neg_rw: float
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


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a specified height off the ground
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w[:, 2]
    standing_position_com_z = asset.data.default_root_state[:, 2]
    standing_height = com_z - standing_position_com_z
    standing_position_toe_roll_z = 0.0626  # recorded from the default position
    offset = (standing_height + standing_position_toe_roll_z).unsqueeze(-1)

    foot_z_target_error = torch.square(
        (
            asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
            - (target_height + offset).repeat(1, 2)
        ).clip(max=0.0)
    )

    # weighted by the velocity of the feet in the xy plane
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_velocity_tanh * foot_z_target_error
    return torch.exp(-torch.sum(reward, dim=1) / std)


@torch.jit.script
def height_target(t: torch.Tensor):
    a5, a4, a3, a2, a1, a0 = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]
    return (a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0)


def track_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """"""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    phase = env.get_phase()

    sin_pos = torch.sin(2 * torch.pi * phase)
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = sin_pos >= 0
    stance_mask[:, 1] = sin_pos < 0
    stance_mask[torch.abs(sin_pos) < 0.1] = 1
    mask_2 = 1 - stance_mask
    mask_2[torch.abs(sin_pos) < 0.1] = 1

    if (torch.sum(contacts == stance_mask) > torch.sum(contacts == mask_2)):
        swing_mask = 1 - stance_mask
    else:
        swing_mask = 1 - mask_2

    filt_foot = torch.where(swing_mask == 1, foot_z, torch.zeros_like(foot_z))

    phase_mod = torch.fmod(phase, 0.5)
    feet_z_target = height_target(phase_mod) #+ offset
    feet_z_value = torch.sum(filt_foot, dim=1)

    error = torch.abs(feet_z_value - feet_z_target)
    reward = torch.exp(-error / std**2)

    return reward

def reward_feet_swing(    
    env: ManagerBasedRLEnv,
    swing_period: float,
    sensor_cfg: SceneEntityCfg
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

def reward_sine_reference(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        ref_angle: float = 0.5,
        double_stand_phase: float = 0.5
) -> torch.Tensor:
        phase = env.get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()

        asset: Articulation = env.scene[asset_cfg.name]
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].clone()
        ref_joint_pos = torch.zeros_like(asset.data.joint_pos[:, asset_cfg.joint_ids])
                  
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        ref_joint_pos[:, 0] = ref_angle * sin_pos_l * 1 #+ asset.data.default_joint_pos[:, 1]
        ref_joint_pos[:, 3] = -ref_angle * sin_pos_l * 2 #+ asset.data.default_joint_pos[:, 4]
        ref_joint_pos[:, 4] = ref_angle * sin_pos_l * 1 #+ asset.data.default_joint_pos[:, 5]
        
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        ref_joint_pos[:, 6] = -ref_angle * sin_pos_r * 1 #+ asset.data.default_joint_pos[:, 7]
        ref_joint_pos[:, 9] = ref_angle * sin_pos_r * 2 #+ asset.data.default_joint_pos[:, 10]
        ref_joint_pos[:, 10] = -ref_angle * sin_pos_r * 1 #+ asset.data.default_joint_pos[:, 11]
        
        # Double support phase
        ref_joint_pos[torch.abs(sin_pos) < double_stand_phase] = 0

        diff = joint_pos - ref_joint_pos
        return torch.exp(-2 * torch.norm(diff, dim=1)) # - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)

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

def reward_symmetry(
          env: ManagerBasedRLEnv,
          asset_cfg: SceneEntityCfg,
          std: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    phase = env.get_phase()
    T_not_done = torch.abs(phase - torch.round(phase)) > 0.04  # True if phase is not done
    T_is_done = torch.abs(phase - torch.round(phase)) <= 0.04  # True if phase is done

    error = torch.norm(env.cycle_torque_diff, dim=-1)
    reward = torch.exp(-error / std**2) * T_is_done
    env.cycle_torque_diff = (env.cycle_torque_diff + 
    torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids[:6]]) - torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids[6:]]) * T_not_done.unsqueeze(1)
    )
    return reward

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