# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections.abc import Sequence
# IsaacSim core
import omni.isaac.core.utils.stage as stage_utils
# IsaacLab core
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

# macros 
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"

##
# Pre-defined configs
##

from isaaclab_tasks.direct.hector.common.robot_core import RobotCore
from isaaclab_tasks.direct.hector.common.rft import PoppySeedCPCfg, PoppySeedLPCfg, RFT_EMF
from isaaclab_tasks.direct.hector.common.contact_point import create_foot_contact_links
from isaaclab_tasks.direct.hector.common.sampler import UniformLineSampler, UniformCubicSampler, GridCubicSampler, QuaternionSampler
from isaaclab_tasks.direct.hector.common.curriculum import CurriculumRateSampler, CurriculumUniformLineSampler, CurriculumUniformCubicSampler, CurriculumQuaternionSampler
from isaaclab_tasks.direct.hector.common.visualization_marker import ContactVisualizer, PenetrationVisualizer

# env config
from isaaclab_tasks.direct.hector.tasks_cfg.soft_terrain_cfg import SoftTerrainEnvCfg

# env base class
from isaaclab_tasks.direct.hector.tasks.base_arch import BaseArch
    

class SoftTerrainEnv(BaseArch):
    cfg: SoftTerrainEnvCfg
    
    def __init__(self, cfg: SoftTerrainEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.curriculum_idx = np.zeros(self.num_envs)
        # RFT
        num_legs = 2
        self._foot_ids = torch.arange(len(self._robot_api.body_names)-num_legs*self.cfg.foot_patch_num, len(self._robot_api.body_names), dtype=torch.long, device=self.device) # foot ids
        self.foot_velocity = torch.zeros(self.num_envs, num_legs*self.cfg.foot_patch_num, 3, device=self.device, dtype=torch.float32)
        self.foot_accel = torch.zeros(self.num_envs, num_legs*self.cfg.foot_patch_num, 3, device=self.device, dtype=torch.float32)
        self.foot_depth = torch.zeros(self.num_envs, num_legs*self.cfg.foot_patch_num, device=self.device, dtype=torch.float32)
        self.foot_angle_beta = torch.zeros(self.num_envs, num_legs*self.cfg.foot_patch_num, device=self.device, dtype=torch.float32)
        self.foot_angle_gamma = torch.zeros(self.num_envs, num_legs*self.cfg.foot_patch_num, device=self.device, dtype=torch.float32)
        self.rft_force = torch.zeros(self.num_envs, num_legs*self.cfg.foot_patch_num, 3, device=self.device, dtype=torch.float32)
    
    def _setup_scene(self)->None:
        """
        Environment specific setup.
        Setup the robot, terrain, sensors, and lights in the scene.
        """
        # robot
        foot_surface = 0.1*0.15
        num_leg = 2
        nx = 2
        ny = 2
        self.cfg.foot_patch_num = nx*ny
        self.cfg.robot.reference_frame = "local"
        self._robot = Articulation(self.cfg.robot)
        stage = stage_utils.get_current_stage()
        create_foot_contact_links(stage, "/World/envs/env_0/Robot", nx, ny, "L")
        create_foot_contact_links(stage, "/World/envs/env_0/Robot", nx, ny, "R")
        self._robot_api = RobotCore(self._robot, num_leg*self.cfg.foot_patch_num)
        self.scene.articulations["robot"] = self._robot
        
        # sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # self._ray_caster = RayCaster(self.cfg.ray_caster)
        # self.scene.sensors["ray_caster"] = self._ray_caster
        
        # visual terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # visualizer markers
        self._contact_visualizer = ContactVisualizer("/Visuals/contact_visualizer")
        self._penetration_visualizer = PenetrationVisualizer("/Visuals/penetration_visualizer")
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # light
        self._light = sim_utils.spawn_light(self.cfg.light.prim_path, self.cfg.light.spawn, orientation=(0.8433914, 0.0, 0.5372996, 0.0))
        
        # RFT
        # dumping_coef=[20, 50, 100]
        self._rft = RFT_EMF(
            # PoppySeedLPCfg(), # loosely packed poppy seed
            PoppySeedCPCfg(), # closely packed poppy seed
            self.device, 
            self.num_envs, 
            num_leg,
            self.cfg.foot_patch_num, 
            foot_surface,
            dynamic_friction_coef=[0.3, 0.4],
            dumping_coef=[2, 10, 1],
            # dumping_coef=[20, 50, 100], # cp
            # dumping_coef=[30, 70, 100],
            )
    
    def _compute_external_perturbation(self)->None:
        """
        Apply RFT lift and drag force
        """
        self.foot_pos = self._robot_api.foot_pos
        self.foot_rot_mat = self._robot_api.foot_rot_mat
        self.foot_depth = self._robot_api.foot_depth
        
        self.foot_velocity = self._robot_api.foot_vel
        self.foot_accel = self._robot_api.foot_accel
        self.foot_angle_beta = self._robot_api.foot_beta_angle
        self.foot_angle_gamma = self._robot_api.foot_gamma_angle
        
        self.rft_force = self._rft.get_force(
            self.foot_depth, 
            self.foot_rot_mat,
            self.foot_velocity,
            self.foot_angle_beta, 
            self.foot_angle_gamma, 
            self._gait_contact) # (num_envs, self.cfg.foot_patch_num, 3)
        self._robot_api.set_external_force(self.rft_force, self._foot_ids, self._robot._ALL_INDICES)
        
        # visualize contact force
        if self.common_step_counter % (self.cfg.rendering_interval//self.cfg.decimation) == 0:
            self._contact_visualizer.visualize(self.foot_pos, self.foot_rot_mat, self.rft_force)
            self._penetration_visualizer.visualize(self.foot_pos, self.foot_rot_mat)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        RL control step
        Process RL output.
        """
        self._actions = actions.clone()
        
        # clip action to (-1, 1) (NOTE: rsl_rl does no thave activation at the last layer)
        if self.cfg.clip_action:
            self._actions = torch.tanh(self._actions)
        # rescale action to custom bounds
        if self.cfg.scale_action:
            positive_mask = (self._actions > 0).to(torch.float32)
            self._actions_op = positive_mask * self.action_ub * self._actions + (1-positive_mask) * self.action_lb * (-self._actions)
        else:
            self._actions_op = self._actions.clone()
        
        # update reference and state of mpc controller
        self._update_mpc_input()
    
    def _run_mpc(self)->None:
        """
        Run MPC and update GRFM and contact state.
        MPC runs at every dt*mpc_decimation (200Hz)
        """
        accel_gyro = []
        grfm = []
        gait_contact = []
        swing_phase = []
        reibert_fps = []
        
        self._get_state()
        for i in range(len(self.mpc)):
            self.mpc[i].set_swing_parameters(stepping_frequency=self._gait_stepping_frequency[i], foot_height=self._foot_height[i])
            self.mpc[i].add_foot_placement_residual(self._foot_placement_residuals[i])
            self.mpc[i].set_srbd_residual(A_residual=self._A_residual[i], B_residual=self._B_residual[i])
            self.mpc[i].update_state(self._state[i].cpu().numpy())
            self.mpc[i].run()
            
            accel_gyro.append(self.mpc[i].accel_gyro(self._root_rot_mat[i].cpu().numpy()))
            grfm.append(self.mpc[i].grfm)
            gait_contact.append(self.mpc[i].contact_state)
            swing_phase.append(self.mpc[i].swing_phase)
            reibert_fps.append(self.mpc[i].reibert_foot_placement)
        
        self._accel_gyro_mpc = torch.from_numpy(np.array(accel_gyro)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self._grfm_mpc = torch.from_numpy(np.array(grfm)).to(self.device).view(self.num_envs, 12).to(torch.float32)
        self._gait_contact = torch.from_numpy(np.array(gait_contact)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self._swing_phase = torch.from_numpy(np.array(swing_phase)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self._reibert_fps = torch.from_numpy(np.array(reibert_fps)).to(self.device).view(self.num_envs, 4).to(torch.float32)
    
    def _split_action(self, policy_action:torch.Tensor)->tuple:
        """
        Split policy action into centroidal acceleration and foot height.
        """
        centroidal_acceleration = policy_action[:, :3]
        centroidal_ang_acceleration = policy_action[:, 3:6]
        return centroidal_acceleration, centroidal_ang_acceleration
    
    def _apply_action(self)->None:
        """
        Actuation control loop
        **********************
        This is kind of like motor actuation loop.
        It is actually not applying self._action to articulation, 
        but rather setting joint effort target.
        And this effort target is passed to actuator model to get dof torques.
        Finally, env.step method calls write_data_to_sim() to write torque to articulation.
        """
        
        # process RFT
        self._compute_external_perturbation()
        
        # process rl actions
        centroidal_accel, centroidal_ang_accel = self._split_action(self._actions_op)
        
        # transform from local to global frame
        centroidal_accel = torch.bmm(self._root_rot_mat, centroidal_accel.unsqueeze(-1)).squeeze(-1)
        centroidal_ang_accel = torch.bmm(self._root_rot_mat, centroidal_ang_accel.unsqueeze(-1)).squeeze(-1)
        self._A_residual[:, 6:9, -1] = centroidal_accel.cpu().numpy()
        self._A_residual[:, 9:12, -1] = centroidal_ang_accel.cpu().numpy()
        
        # run mpc controller
        self._run_mpc()
        
        # run low level control with updated GRFM
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
    
    def _get_observations(self) -> dict:
        """
        Get actor and critic observations.
        """
        self._previous_actions = self._actions.clone()
        self._get_contact_observation()
        self._obs = torch.cat(
            (
                self._root_pos[:, 2:], #0:1 (only height)
                self._root_quat, #1:5
                self._root_lin_vel_b, #5:8
                self._root_ang_vel_b, #8:11
                self._desired_root_lin_vel_b, #11:13
                self._desired_root_ang_vel_b, #13:14
                self._joint_pos, #14:24
                self._joint_vel, #24:34
                self._joint_effort, #34:44
                self._previous_actions, #44:50
                self._accel_gyro_mpc, #50:56
                self._gait_contact, #56:58
                self._swing_phase, #58:60
            ),
            dim=-1,
        )
        observation = {"policy": self._obs}
        return observation
    
    def _reset_idx(self, env_ids: Sequence[int])->None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        print(f"[INFO] Reset environment {env_ids} at step {self.episode_length_buf[env_ids]}")
        super()._reset_idx(env_ids)
        
        self._reset_robot(env_ids)
        
        # reset terrain parameters if one of envs is reset
        self._reset_terrain(env_ids)
        
        # logging
        if "log" not in self.extras:
            self.extras["log"] = dict()
        for key, value in self.episode_sums.items():
            self.extras["log"].update({f"Reward_episode/{key}": value})
            self.episode_sums[key] = 0.0
    
    def _reset_terrain(self, env_ids: Sequence[int])->None:
        pass
        
    
    def _reset_robot(self, env_ids: Sequence[int])->None:
        # num_curriculum_x = self.cfg.terrain.terrain_generator.num_cols
        # num_curriculum_y = self.cfg.terrain.terrain_generator.num_rows
        # curriculum_idx = np.floor(self.cfg.terrain_curriculum_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)))
        # self.curriculum_idx = curriculum_idx
        # center_coord = np.stack([self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx//num_curriculum_x), 
        #                          self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx%num_curriculum_y)], axis=-1)
        
        center_coord = np.array([self.cfg.terrain.center_position[0], self.cfg.terrain.center_position[1]])[None, :]
        # position = self.cfg.robot_position_sampler.sample(center_coord, len(env_ids))

        position = self.cfg.robot_position_sampler.sample(len(env_ids))
        quat = self.cfg.robot_quat_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(position))
        
        position = torch.tensor(position, device=self.device).view(-1, 3)
        quat = torch.tensor(quat, device=self.device).view(-1, 4)
        default_root_pose = torch.cat((position, quat), dim=-1)
        
        # override the default state
        self._robot_api.reset_default_pose(default_root_pose, env_ids)
        
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = self._robot_api.default_root_state[env_ids, 7:]
        
        # reset joint position
        joint_pos = self._robot_api.default_joint_pos[:, self._joint_ids][env_ids]
        joint_vel = self._robot_api.default_joint_vel[:, self._joint_ids][env_ids]
        self._joint_pos[env_ids] = joint_pos
        self._joint_vel[env_ids] = joint_vel
        self._add_joint_offset(env_ids)
        
        # write to sim
        self._robot_api.write_root_pose_to_sim(default_root_pose, env_ids)
        self._robot_api.write_root_velocity_to_sim(default_root_vel, env_ids)
        self._robot_api.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_ids, env_ids)
        
        # reset mpc reference
        self._desired_twist_np[env_ids.cpu().numpy()] = np.array(self.cfg.robot_target_velocity_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32)
        self.mpc_ctrl_counter[env_ids] = 0
        for i in env_ids.cpu().numpy():
            self.mpc[i].reset()
        
        # reset reference 
        self._ref_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._ref_yaw = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        
        # update view port to look at the current active terrain
        self.viewport_camera_controller.update_view_location(eye=(center_coord[0, 0]+4, center_coord[0, 1]-4, 1.0), lookat=(center_coord[0, 0], center_coord[0, 1], 0.0))
    
    def _get_rewards(self)->torch.Tensor:
        # reward
        self.height_reward, self.lin_vel_reward, self.ang_vel_reward = \
            self.cfg.reward_parameter.compute_reward(
            self._root_pos,
            self._root_lin_vel_b,
            self._root_ang_vel_b,
            self.cfg.reference_height,
            self._desired_root_lin_vel_b,
            self._desired_root_ang_vel_b,
        )
        
        self.position_reward, self.yaw_reward = self.cfg.pose_tracking_reward_parameter.compute_reward(
            self._root_pos[:, :2], # x, y
            self._root_yaw,
            self._ref_pos[:, :2], # x, y
            self._ref_yaw
        )
            
        self.alive_reward = self.cfg.alive_reward_parameter.compute_reward(self.reset_terminated, self.episode_length_buf, self.max_episode_length)
        
        # penalty
        self.roll_penalty, self.pitch_penalty, self.action_penalty, self.energy_penalty, self.foot_energy_penalty = \
            self.cfg.penalty_parameter.compute_penalty(
            self._root_quat,
            self._actions,
            self._previous_actions
        )
            
        self.vx_penalty, self.vy_penalty, self.wz_penalty = \
            self.cfg.twist_penalty_parameter.compute_penalty(
            self._root_lin_vel_b,
            self._root_ang_vel_b
            )
        
        self.foot_slide_penalty = self.cfg.foot_slide_penalty_parameter.compute_penalty(self._robot_api.body_lin_vel_w[:, -2:], self._gt_contact) # ankle at last 2 body index
        self.action_saturation_penalty = self.cfg.action_saturation_penalty_parameter.compute_penalty(self._actions)
        
        # scale rewards and penalty with time step (following reward manager in manager based rl env)
        self.height_reward = self.height_reward * self.step_dt
        self.lin_vel_reward = self.lin_vel_reward * self.step_dt
        self.ang_vel_reward = self.ang_vel_reward * self.step_dt
        self.alive_reward = self.alive_reward * self.step_dt
        self.position_reward = self.position_reward * self.step_dt
        self.yaw_reward = self.yaw_reward * self.step_dt
        
        self.roll_penalty = self.roll_penalty * self.step_dt
        self.pitch_penalty = self.pitch_penalty * self.step_dt
        self.action_penalty = self.action_penalty * self.step_dt
        self.energy_penalty = self.energy_penalty * self.step_dt
        self.foot_energy_penalty = self.foot_energy_penalty * self.step_dt
        self.vx_penalty = self.vx_penalty * self.step_dt
        self.vy_penalty = self.vy_penalty * self.step_dt
        self.wz_penalty = self.wz_penalty * self.step_dt
        self.foot_slide_penalty = self.foot_slide_penalty * self.step_dt
        self.action_saturation_penalty = self.action_saturation_penalty * self.step_dt
         
        reward = self.height_reward + self.lin_vel_reward + self.ang_vel_reward + self.alive_reward + \
            self.position_reward + self.yaw_reward
        penalty = self.roll_penalty + self.pitch_penalty + self.action_penalty + self.energy_penalty + \
            self.foot_energy_penalty + self.vx_penalty + self.vy_penalty + self.wz_penalty + \
            self.foot_slide_penalty + self.action_saturation_penalty
         
        total_reward = reward - penalty
        
        # push logs to extras
        self.extras["log"] = dict()
        self.log_state()
        self.log_action()
        self.log_curriculum()
        self.log_episode_reward()
            
        return total_reward
    
    def _get_dones(self)->tuple[torch.Tensor, torch.Tensor]:
        # timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        roll = torch.atan2(2*(self._root_quat[:, 0]*self._root_quat[:, 1] + self._root_quat[:, 2]*self._root_quat[:, 3]), 1 - 2*(self._root_quat[:, 1]**2 + self._root_quat[:, 2]**2))
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        
        pitch = torch.asin(2*(self._root_quat[:, 0]*self._root_quat[:, 2] - self._root_quat[:, 3]*self._root_quat[:, 1]))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        
        # base angle and base height violation
        roll_reset = torch.abs(roll) > self.cfg.roll_limit
        pitch_reset = torch.abs(pitch) > self.cfg.pitch_limit
        height_reset = (self._root_pos[:, 2] < self.cfg.min_height) | (self._root_pos[:, 2] > self.cfg.max_height)
        
        reset = roll_reset | pitch_reset
        reset = reset | height_reset
        
        return reset, time_out
    
    ### specific to the architecture ###
    def log_episode_reward(self)->None:
        self.episode_sums["height_reward"] += self.height_reward[0].item()
        self.episode_sums["lin_vel_reward"] += self.lin_vel_reward[0].item()
        self.episode_sums["ang_vel_reward"] += self.ang_vel_reward[0].item()
        self.episode_sums["alive_reward"] += self.alive_reward[0].item()
        self.episode_sums["position_reward"] += self.position_reward[0].item()
        self.episode_sums["yaw_reward"] += self.yaw_reward[0].item()
        
        self.episode_sums["roll_penalty"] += self.roll_penalty[0].item()
        self.episode_sums["pitch_penalty"] += self.pitch_penalty[0].item()
        self.episode_sums["action_penalty"] += self.action_penalty[0].item()
        self.episode_sums["energy_penalty"] += self.energy_penalty[0].item()
        self.episode_sums["foot_energy_penalty"] += self.foot_energy_penalty[0].item()
        self.episode_sums["vx_penalty"] += self.vx_penalty[0].item()
        self.episode_sums["vy_penalty"] += self.vy_penalty[0].item()
        self.episode_sums["wz_penalty"] += self.wz_penalty[0].item()
        self.episode_sums["feet_slide_penalty"] += self.foot_slide_penalty[0].item()
        self.episode_sums["action_saturation_penalty"] += self.action_saturation_penalty[0].item()
    
    def log_curriculum(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            curriculum_idx = self.curriculum_idx[0]
            log["curriculum/curriculum_idx"] = curriculum_idx
        self.extras["log"].update(log)
        
    def log_state(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            root_pos = self._root_pos[0].cpu().numpy()
            root_lin_vel_b = self._root_lin_vel_b[0].cpu().numpy()
            root_ang_vel_b = self._root_ang_vel_b[0].cpu().numpy()
            desired_root_lin_vel_b = self._desired_root_lin_vel_b[0].cpu().numpy()
            desired_root_ang_vel_b = self._desired_root_ang_vel_b[0].cpu().numpy()
            mpc_centroidal_accel = self._accel_gyro_mpc[0, :3].cpu().numpy()
            mpc_centroidal_ang_accel = self._accel_gyro_mpc[0, 3:].cpu().numpy()
            log["state/root_pos_x"] = root_pos[0]
            log["state/root_pos_y"] = root_pos[1]
            log["state/root_pos_z"] = root_pos[2]
            log["state/root_lin_vel_x"] = root_lin_vel_b[0]
            log["state/root_lin_vel_y"] = root_lin_vel_b[1]
            log["state/root_ang_vel_z"] = root_ang_vel_b[2]
            log["state/desired_root_lin_vel_x"] = desired_root_lin_vel_b[0]
            log["state/desired_root_lin_vel_y"] = desired_root_lin_vel_b[1]
            log["state/desired_root_ang_vel_z"] = desired_root_ang_vel_b[0]
            log["state/centroidal_accel_x"] = mpc_centroidal_accel[0]
            log["state/centroidal_accel_y"] = mpc_centroidal_accel[1]
            log["state/centroidal_accel_z"] = mpc_centroidal_accel[2]
            log["state/centroidal_ang_accel_x"] = mpc_centroidal_ang_accel[0]
            log["state/centroidal_ang_accel_y"] = mpc_centroidal_ang_accel[1]
            log["state/centroidal_ang_accel_z"] = mpc_centroidal_ang_accel[2]
        self.extras["log"].update(log)
    
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            centroidal_acceleration = self._actions[0, :3].cpu().numpy()
            centroidal_ang_acceleration = self._actions[0, 3:6].cpu().numpy()
            log["raw_action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["raw_action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["raw_action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["raw_action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["raw_action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["raw_action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
            
            centroidal_acceleration = self._actions_op[0, :3].cpu().numpy()
            centroidal_ang_acceleration = self._actions_op[0, 3:6].cpu().numpy()
            log["action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
        self.extras["log"].update(log)