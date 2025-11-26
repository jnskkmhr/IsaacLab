# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils.string import resolve_matching_names_values
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    """
    Finds the mapping between PhysX and MJWarp joint names.
    Returns a tuple of two lists: (mjc_to_physx, physx_to_mjc).
    """
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc
    
"""
whole-body centroidal momentum penalties.
"""

class angular_momentum_l2(ManagerTermBase):
    """
    compute the L2 norm of the whole-body centroidal (pelvis) angular momentum.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        import sys 
        # TODO: figure out non sys path way to import cusadi
        sys.path.append("/home/jkamohara3/isaac/debug")
        import casadi
        from cusadi import CASADI_FUNCTION_DIR, CusadiFunction

        super().__init__(cfg, env)
        self.centroidal_ang_momentum = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.device)
        self.cusadi_func = CusadiFunction(
            casadi.Function.load(os.path.join(CASADI_FUNCTION_DIR, "g1_29dof_ang_momentum_func.casadi")),  # type: ignore
            num_instances=env.num_envs, 
            )
        
        # get joint mapping index 
        assert len(cfg.params["physx_joint_names"]) == len(cfg.params["mjw_joint_names"]), "PhysX and MJWarp joint name lists must have the same length."
        self.mjc_to_physx, self.physx_to_mjc = find_physx_mjwarp_mapping(cfg.params["mjw_joint_names"], cfg.params["physx_joint_names"])
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        physx_joint_names: list[str],
        mjw_joint_names: list[str],
    ) -> torch.Tensor:
        
        asset: Articulation = env.scene[asset_cfg.name]
        base_pos = asset.data.root_pos_w - env.scene.env_origins # (num_envs, 3)
        base_quat = asset.data.root_quat_w # (num_envs, 4)
        # align physx joint order to mjw order
        joint_pos = asset.data.joint_pos.clone()[:, self.mjc_to_physx] # (num_envs, num_dofs)

        base_lin_vel = asset.data.root_lin_vel_w # (num_envs, 3)
        base_ang_vel = asset.data.root_ang_vel_w # (num_envs, 3)
        # align physx joint order to mjw order
        joint_vel = asset.data.joint_vel.clone()[:, self.mjc_to_physx] # (num_envs, num_dofs)

        q_pos = torch.cat([base_pos, base_quat, joint_pos], dim=-1)
        q_vel = torch.cat([base_lin_vel, base_ang_vel, joint_vel], dim=-1)
        self.cusadi_func.evaluate([q_pos.double(), q_vel.double()])

        # whole body centroidal angular momentum wrt global frame
        self.centroidal_ang_momentum = self.cusadi_func.getDenseOutput(0).squeeze(-1).float()
        return torch.sqrt(torch.square(self.centroidal_ang_momentum).sum(dim=-1))


"""
dof regularization penalties.
"""

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward

class variable_posture(ManagerTermBase):
    """
    compute gaussian kernel reward to regularize robot's whole body posture for each gait.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_joint_pos = asset.data.default_joint_pos 

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, std_standing = resolve_matching_names_values(
        data=cfg.params["std_standing"],
        list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(
        std_standing, device=env.device, dtype=torch.float32
        )

        _, _, std_walking = resolve_matching_names_values(
        data=cfg.params["std_walking"],
        list_of_strings=joint_names,
        )
        self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

        _, _, std_running = resolve_matching_names_values(
        data=cfg.params["std_running"],
        list_of_strings=joint_names,
        )
        self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        std_standing: dict, 
        std_walking: dict,
        std_running: dict,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        
        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        linear_speed = torch.norm(command[:, :2], dim=-1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
        (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
        self.std_standing * standing_mask.unsqueeze(1)
        + self.std_walking * walking_mask.unsqueeze(1)
        + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))

"""
gait
"""
def reward_feet_swing(    
    env: ManagerBasedRLEnv,
    swing_period: float,
    sensor_cfg: SceneEntityCfg,
    cmd_threshold: float = 0.05,
    command_name=None,
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

    # # weight by command magnitude
    # if command_name is not None:
    #     cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    #     reward *= cmd_norm > cmd_threshold

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

    # leg_phase = env.get_phase()

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i]) # reward contact match (swing-swing or contact-contact)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > cmd_threshold
    
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    if "log" in env.extras.keys():
        env.extras["log"]["Metrics/feet_air_time"] = air_time.mean()

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
    """Reward the swinging feet for clearing a specified height off the ground, weighted by foot velocity."""
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