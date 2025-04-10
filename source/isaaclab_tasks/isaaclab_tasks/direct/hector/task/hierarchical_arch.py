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

# IsaacLab core
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, RayCaster

# macros 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"

##
# Pre-defined configs
##

# Task core
from isaaclab_tasks.direct.hector.common.robot_core import RobotCore
from  isaaclab_tasks.direct.hector.common.utils.data_util import HistoryBuffer
from isaaclab_tasks.direct.hector.common.visualization_marker import FootPlacementVisualizer, VelocityVisualizer, SwingFootVisualizer

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.hierarchical_arch_cfg import HierarchicalArchCfg, HierarchicalArchPrimeCfg, HierarchicalArchAccelPFCfg, HierarchicalArchPrimeFullCfg

# Base class
from isaaclab_tasks.direct.hector.task.base_arch import BaseArch
    

class HierarchicalArch(BaseArch):
    cfg: HierarchicalArchCfg
    
    def __init__(self, cfg: HierarchicalArchCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.curriculum_idx = np.zeros(self.num_envs)
        
        # for logging
        self.episode_reward_sums = {
            "height_reward": torch.zeros(self.num_envs, device=self.device),
            "lin_vel_reward": torch.zeros(self.num_envs, device=self.device),
            "ang_vel_reward": torch.zeros(self.num_envs, device=self.device),
            "alive_reward": torch.zeros(self.num_envs, device=self.device),
            "contact_reward": torch.zeros(self.num_envs, device=self.device),
            "position_reward": torch.zeros(self.num_envs, device=self.device),
            "yaw_reward": torch.zeros(self.num_envs, device=self.device),
            "swing_foot_tracking_reward": torch.zeros(self.num_envs, device=self.device),
        }
        
        self.episode_penalty_sums = {
            "velocity_penalty": torch.zeros(self.num_envs, device=self.device), # this is a velocity penalty for the root linear velocity
            "ang_velocity_penalty": torch.zeros(self.num_envs, device=self.device), # angular velocity penalty for the root angular velocity
            "feet_slide_penalty": torch.zeros(self.num_envs, device=self.device),
            "hip_pitch_deviation_penalty": torch.zeros(self.num_envs, device=self.device),
            "foot_distance_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_saturation_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_penalty": torch.zeros(self.num_envs, device=self.device),
            "energy_penalty": torch.zeros(self.num_envs, device=self.device),
            "torque_penalty": torch.zeros(self.num_envs, device=self.device),
        }
    
    def _setup_scene(self)->None:
        """
        Environment specific setup.
        Setup the robot, terrain, sensors, and lights in the scene.
        """
        # robot
        self._robot = Articulation(self.cfg.robot)
        self._robot_api = RobotCore(self._robot, self.num_envs, self.cfg.foot_patch_num)
        self.scene.articulations["robot"] = self._robot
        
        # sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        # ray caster
        self._raycaster = RayCaster(self.cfg.ray_caster)
        self.scene.sensors["raycaster"] = self._raycaster
        
        # base terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # visualization marker
        self.foot_placement_visualizer = FootPlacementVisualizer("/Visuals/foot_placement")
        self._velocity_visualizer = VelocityVisualizer("/Visuals/velocity_visualizer")
        self.swing_foot_visualizer = SwingFootVisualizer("/Visuals/swing_foot_visualizer")
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # light
        self._light = sim_utils.spawn_light(self.cfg.light.prim_path, self.cfg.light.spawn, orientation=(0.8433914, 0.0, 0.5372996, 0.0))
        
        # update viewer
        curriculum_idx = 0
        num_curriculum_x = int(self.cfg.terrain.terrain_generator.num_cols/self.cfg.terrain.friction_group_patch_num)
        num_curriculum_y = int(self.cfg.terrain.terrain_generator.num_rows/self.cfg.terrain.friction_group_patch_num)
        terrain_origin = np.array([self.cfg.terrain.center_position[0], self.cfg.terrain.center_position[1], 0.0])
        camera_pos = terrain_origin + np.array([self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx//num_curriculum_x), 
                                 self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx%num_curriculum_y), 0])
        camera_delta = np.array([0.0, -4.0, 0.0])
        self.cfg.viewer.eye = (camera_pos[0]+camera_delta[0], camera_pos[1]+camera_delta[1], camera_pos[2]+camera_delta[2])
        self.cfg.viewer.lookat = (camera_pos[0], camera_pos[1], 0.0)
    
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
        
        if self.cfg.curriculum_inference:
            # update reference and state of mpc controller
            env_ids = self._robot._ALL_INDICES
            robot_twist = np.array(self.cfg.robot_target_velocity_sampler.sample(self.common_step_counter//self.cfg.num_steps_per_env, len(env_ids)), dtype=np.float32)
            self._desired_twist_np[env_ids.cpu().numpy()] = robot_twist # type: ignore
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
        augmented_fps = []
        
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
            augmented_fps.append(self.mpc[i].foot_placement)
        
        self._accel_gyro_mpc = torch.from_numpy(np.array(accel_gyro)).to(self.device).view(self.num_envs, 6).to(torch.float32)
        self._grfm_mpc = torch.from_numpy(np.array(grfm)).to(self.device).view(self.num_envs, 12).to(torch.float32)
        self._gait_contact = torch.from_numpy(np.array(gait_contact)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self._swing_phase = torch.from_numpy(np.array(swing_phase)).to(self.device).view(self.num_envs, 2).to(torch.float32)
        self._reibert_fps = torch.from_numpy(np.array(reibert_fps)).to(self.device).view(self.num_envs, 4).to(torch.float32)
        self._augmented_fps = torch.from_numpy(np.array(augmented_fps)).to(self.device).view(self.num_envs, 4).to(torch.float32)
    
    def _split_action(self, policy_action:torch.Tensor)->torch.Tensor:
        """
        Split policy action into centroidal acceleration,
        """
        centroidal_acceleration = policy_action[:, :3]
        return centroidal_acceleration
    
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
        # process rl actions
        centroidal_accel = self._split_action(self._actions_op)
        # centroidal_accel = torch.bmm(self._root_rot_mat, centroidal_accel.unsqueeze(-1)).squeeze(-1)
        self._A_residual[:, 6:9, -1] = centroidal_accel.cpu().numpy()
        
        # run mpc controller
        self._run_mpc()
        
        # run low level control with updated GRFM
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
        
        self.visualize_marker()
    
    def visualize_marker(self):
        if self.common_step_counter % (self.cfg.rendering_interval/self.cfg.decimation) == 0:
            reibert_fps = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
            augmented_fps = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
            default_position = self._robot_api.default_root_state[:, :3]
            default_position[:, 2] = self._robot_api.root_pos_w[:, 2] - self._root_pos[:, 2]
            reibert_fps[:, 0, :2] = self._reibert_fps[:, :2]
            reibert_fps[:, 1, :2] = self._reibert_fps[:, 2:]
            augmented_fps[:, 0, :2] = self._augmented_fps[:, :2]
            augmented_fps[:, 1, :2] = self._augmented_fps[:, 2:]
            
            # convert local foot placement to simulation global frame
            reibert_fps[:, 0, :] = torch.bmm(self._init_rot_mat, reibert_fps[:, 0, :].unsqueeze(-1)).squeeze(-1) + default_position
            reibert_fps[:, 1, :] = torch.bmm(self._init_rot_mat, reibert_fps[:, 1, :].unsqueeze(-1)).squeeze(-1) + default_position 
            augmented_fps[:, 0, :] = torch.bmm(self._init_rot_mat, augmented_fps[:, 0, :].unsqueeze(-1)).squeeze(-1) + default_position
            augmented_fps[:, 1, :] = torch.bmm(self._init_rot_mat, augmented_fps[:, 1, :].unsqueeze(-1)).squeeze(-1) + default_position
            
            # hide foot placement marker when foot is in contact
            reibert_fps[:, 0, 2] -= self._gait_contact[:, 0] * 5.0
            reibert_fps[:, 1, 2] -= self._gait_contact[:, 1] * 5.0
            augmented_fps[:, 0, 2] -= self._gait_contact[:, 0] * 5.0
            augmented_fps[:, 1, 2] -= self._gait_contact[:, 1] * 5.0
            
            
            # swing foot
            left_swing = (self._root_rot_mat @ self._ref_foot_pos_b[:, :3].unsqueeze(2)).squeeze(2) + self._root_pos
            right_swing = (self._root_rot_mat @ self._ref_foot_pos_b[:, 3:].unsqueeze(2)).squeeze(2) + self._root_pos
            
            left_swing = (self._init_rot_mat @ left_swing.unsqueeze(-1)).squeeze(-1) + default_position
            right_swing = (self._init_rot_mat @ right_swing.unsqueeze(-1)).squeeze(-1) + default_position
            swing_reference = torch.stack((left_swing, right_swing), dim=1)
            
            orientation = self._robot_api.root_quat_w.repeat(4, 1)
            
            self.foot_placement_visualizer.visualize(reibert_fps, augmented_fps, orientation)
            self._velocity_visualizer.visualize(self._robot_api.root_pos_w, self._robot_api.root_quat_w, self._robot_api.root_lin_vel_b)
            self.swing_foot_visualizer.visualize(swing_reference)
    
    def _get_observations(self) -> dict:
        """
        Get actor and critic observations.
        """
        self._previous_actions = self._actions.clone()
        self._get_contact_observation()
        self._obs = torch.cat(
            (
                self._root_pos[:, 2:], #0:1 z
                self._root_quat, #1:5
                self._root_lin_vel_b, #5:8
                self._root_ang_vel_b, #8:11
                self._desired_root_lin_vel_b, #11:13
                self._desired_root_ang_vel_b, #13:14
                self._joint_pos, #14:24
                self._joint_vel, #24:34
                self._joint_effort, #34:44
                self._previous_actions, #44:47
                self._accel_gyro_mpc[:, :3], #47:50
                self._gait_contact, #50:52
                self._gt_contact, #52:54
            ),
            dim=-1,
        )
        observation = {"policy": self._obs}
        return observation
    
    def _reset_idx(self, env_ids: Sequence[int])->None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # print(f"[INFO] Reset environment {env_ids} at step {self.episode_length_buf[env_ids]}")
        # print("[INFO] Robot desired velocity: ", self._desired_twist_np.tolist())
        super()._reset_idx(env_ids)
        
        self._reset_robot(env_ids)
        self._reset_terrain(env_ids)
        
        # log
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.log_curriculum()
        
        # log episode reward
        for key in self.episode_reward_sums.keys():
            episode_sums = torch.mean(self.episode_reward_sums[key][env_ids])
            self.extras["log"].update({f"Episode_reward/{key}": episode_sums/self.max_episode_length_s})
            self.episode_reward_sums[key][env_ids] = 0.0
        for key in self.episode_penalty_sums.keys():
            episode_sums = torch.mean(self.episode_penalty_sums[key][env_ids])
            self.extras["log"].update({f"Episode_penalty/{key}": episode_sums/self.max_episode_length_s})
            self.episode_penalty_sums[key][env_ids] = 0.0
    
    def _reset_terrain(self, env_ids: Sequence[int])->None:
        pass
    
    def _get_sub_terrain_center(self, curriculum_idx:np.ndarray):
        if self.cfg.terrain.terrain_type == "generator":
            # terrain left bottom corner is at the origin
            terrain_size_x = self.cfg.terrain.terrain_generator.size[0]*self.cfg.terrain.terrain_generator.num_cols
            terrain_size_y = self.cfg.terrain.terrain_generator.size[1]*self.cfg.terrain.terrain_generator.num_rows
            
            num_tiles_per_curriculum = self.cfg.terrain.terrain_generator.num_cols// int(math.sqrt(self.cfg.terrain.num_curriculums))
            nx = self.cfg.terrain.terrain_generator.num_cols // num_tiles_per_curriculum # number of sub-terrain in x direction
            ny = self.cfg.terrain.terrain_generator.num_rows // num_tiles_per_curriculum # number of sub-terrain in y direction
            center_coord = \
                np.stack([(self.cfg.terrain.terrain_generator.size[0] * num_tiles_per_curriculum)/2 + \
                    (self.cfg.terrain.terrain_generator.size[0] * num_tiles_per_curriculum)*(curriculum_idx//nx) - terrain_size_x//2, 
                        (self.cfg.terrain.terrain_generator.size[1] * num_tiles_per_curriculum)/2 + \
                            (self.cfg.terrain.terrain_generator.size[1] * num_tiles_per_curriculum)*(curriculum_idx%ny) - terrain_size_y//2], 
                        axis=-1)
        elif self.cfg.terrain.terrain_type == "patched":
            # terrain center is at the origin
            num_curriculum_x = int(self.cfg.terrain.terrain_generator.num_cols/self.cfg.terrain.friction_group_patch_num)
            num_curriculum_y = int(self.cfg.terrain.terrain_generator.num_rows/self.cfg.terrain.friction_group_patch_num)
            
            final_curriculum_mask = self.common_step_counter//self.cfg.num_steps_per_env >= 4000
            curriculum_idx[final_curriculum_mask] = (self.cfg.terrain.num_curriculums - 1) * np.ones_like(curriculum_idx[final_curriculum_mask])
            
            center_coord = np.stack([
                self.cfg.terrain.terrain_generator.size[0]*(self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx//num_curriculum_x)), 
                self.cfg.terrain.terrain_generator.size[1]*(self.cfg.terrain.friction_group_patch_num/2 + self.cfg.terrain.friction_group_patch_num*(curriculum_idx%num_curriculum_y))], 
                                    axis=-1)
        else:
            center_coord = np.zeros((len(curriculum_idx), 2))
        return center_coord
    
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
        
        self.swing_foot_tracking_reward = self.cfg.swing_foot_tracking_reward_parameter.compute_reward(
            (self._foot_pos_b.reshape(-1, 2, 3) * (1-self._gait_contact).reshape(-1, 2, 1)).reshape(-1, 6), 
            (self._ref_foot_pos_b.reshape(-1, 2, 3) * (1-self._gait_contact).reshape(-1, 2, 1)).reshape(-1, 6),
        )
            
        
        # penalty
        self.roll_penalty, self.pitch_penalty = self.cfg.orientation_penalty_parameter.compute_penalty(self._root_quat)
        self.action_penalty, self.energy_penalty = self.cfg.action_penalty_parameter.compute_penalty(
            self._actions, 
            self._previous_actions, 
            self.common_step_counter//self.cfg.num_steps_per_env)
        self.vx_penalty, self.vy_penalty, self.wz_penalty = self.cfg.twist_penalty_parameter.compute_penalty(
            self._root_lin_vel_b,
            self._root_ang_vel_b
            )
        self.velocity_penalty = self.cfg.velocity_penalty_parameter.compute_penalty(self._root_lin_vel_b)
        self.ang_velocity_penalty = self.cfg.angular_velocity_penalty_parameter.compute_penalty(self._root_ang_vel_b)
        
        self.foot_slide_penalty = self.cfg.foot_slide_penalty_parameter.compute_penalty(self._robot_api.body_lin_vel_w[:, -2:, :2], self._gt_contact) # ankle at last 2 body index
        self.action_saturation_penalty = self.cfg.action_saturation_penalty_parameter.compute_penalty(self._actions)
        self.termination_penalty = self.cfg.termination_penalty_parameter.compute_penalty(self.reset_terminated)
        self.foot_distance_penalty = self.cfg.foot_distance_penalty_parameter.compute_penalty(self._foot_pos_b[:, :3], self._foot_pos_b[:, 3:])
        self.torque_penalty = self.cfg.torque_penalty_parameter.compute_penalty(self._joint_actions, self.common_step_counter//self.cfg.num_steps_per_env)
        
        # scale rewards and penalty with time step (following reward manager in manager based rl env)
        self.height_reward = self.height_reward * self.step_dt
        self.lin_vel_reward = self.lin_vel_reward * self.step_dt
        self.ang_vel_reward = self.ang_vel_reward * self.step_dt
        self.alive_reward = self.alive_reward * self.step_dt
        self.position_reward = self.position_reward * self.step_dt
        self.yaw_reward = self.yaw_reward * self.step_dt
        self.swing_foot_tracking_reward = self.swing_foot_tracking_reward * self.step_dt
        
        self.roll_penalty = self.roll_penalty * self.step_dt
        self.pitch_penalty = self.pitch_penalty * self.step_dt
        self.vx_penalty = self.vx_penalty * self.step_dt
        self.vy_penalty = self.vy_penalty * self.step_dt
        self.wz_penalty = self.wz_penalty * self.step_dt

        self.velocity_penalty = self.velocity_penalty * self.step_dt
        self.ang_velocity_penalty = self.ang_velocity_penalty * self.step_dt

        self.action_penalty = self.action_penalty * self.step_dt
        self.energy_penalty = self.energy_penalty * self.step_dt
        self.foot_slide_penalty = self.foot_slide_penalty * self.step_dt
        self.action_saturation_penalty = self.action_saturation_penalty * self.step_dt
        self.foot_distance_penalty = self.foot_distance_penalty * self.step_dt
        self.torque_penalty = self.torque_penalty * self.step_dt
         
        reward = self.height_reward + self.lin_vel_reward + self.ang_vel_reward + self.position_reward + self.yaw_reward + self.swing_foot_tracking_reward + self.alive_reward
        penalty = self.roll_penalty + self.pitch_penalty + self.action_penalty + self.energy_penalty + self.vx_penalty + self.vy_penalty + self.wz_penalty + \
            self.foot_slide_penalty + self.action_saturation_penalty + self.foot_distance_penalty + self.torque_penalty + self.velocity_penalty + self.ang_velocity_penalty
        
        total_reward = reward - penalty
        
        # push logs to extras
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.log_state()
        self.log_action()
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
        self.episode_reward_sums["height_reward"] += self.height_reward[0].item()
        self.episode_reward_sums["lin_vel_reward"] += self.lin_vel_reward[0].item()
        self.episode_reward_sums["ang_vel_reward"] += self.ang_vel_reward[0].item()
        self.episode_reward_sums["alive_reward"] += self.alive_reward[0].item()
        self.episode_reward_sums["position_reward"] += self.position_reward[0].item()
        self.episode_reward_sums["yaw_reward"] += self.yaw_reward[0].item()

        self.episode_penalty_sums["velocity_penalty"] += self.velocity_penalty[0].item() 
        self.episode_penalty_sums["ang_velocity_penalty"] += self.ang_velocity_penalty[0].item() 
        self.episode_penalty_sums["feet_slide_penalty"] += self.foot_slide_penalty[0].item()
        self.episode_penalty_sums["foot_distance_penalty"] += self.foot_distance_penalty[0].item()
        self.episode_penalty_sums["action_penalty"] += self.action_penalty[0].item()
        self.episode_penalty_sums["energy_penalty"] += self.energy_penalty[0].item()
        self.episode_penalty_sums["action_saturation_penalty"] += self.action_saturation_penalty[0].item()
        self.episode_penalty_sums["torque_penalty"] += self.torque_penalty[0].item()
    
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
            # raw action
            centroidal_acceleration = self._actions[0, :3].cpu().numpy()
            log["raw_action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["raw_action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["raw_action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            
            # clipped action
            centroidal_acceleration = self._actions_op[0, :3].cpu().numpy()
            log["action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["action/centroidal_acceleration_z"] = centroidal_acceleration[2]
        self.extras["log"].update(log)



class HierarchicalArchPrime(HierarchicalArch):
    cfg: HierarchicalArchPrimeCfg
    
    def __init__(self, cfg: HierarchicalArchCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
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
        
        self.visualize_marker()
    
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
    
    ### specific to the architecture ###
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


class HierarchicalArchPrimeFull(HierarchicalArch):
    cfg: HierarchicalArchPrimeFullCfg
    def __init__(self, cfg: HierarchicalArchPrimeFullCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.num_history = self.cfg.num_history
        self.history_buffer = HistoryBuffer(self.num_envs, self.num_history, self.cfg.observation_space, torch.float32, self.device)
        
    def _split_action(self, policy_action:torch.Tensor)->tuple:
        """
        Split policy action into centroidal acceleration and foot height.
        """
        centroidal_acceleration = policy_action[:, :3]
        centroidal_ang_acceleration = policy_action[:, 3:6]
        added_mass_inv1 = policy_action[:, 6:9]
        added_mass_inv2 = policy_action[:, 9:12]
        added_inertia_inv1 = policy_action[:, 12:15]
        added_inertia_inv2 = policy_action[:, 15:18]
        return centroidal_acceleration, centroidal_ang_acceleration, added_mass_inv1, added_mass_inv2, added_inertia_inv1, added_inertia_inv2
    
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
        # process rl actions
        centroidal_accel, centroidal_ang_accel, added_mass_inv1, added_mass_inv2, added_inertia_inv1, added_inertia_inv2 = self._split_action(self._actions_op)

        # form residual dynamics matrix
        centroidal_accel = torch.bmm(self._root_rot_mat, centroidal_accel.unsqueeze(-1)).squeeze(-1)
        centroidal_ang_accel = torch.bmm(self._root_rot_mat, centroidal_ang_accel.unsqueeze(-1)).squeeze(-1)
        self._A_residual[:, 6:9, -1] = centroidal_accel.cpu().numpy()
        self._A_residual[:, 9:12, -1] = centroidal_ang_accel.cpu().numpy()
        self._B_residual[:, 6:9, 6:9] = torch.diag_embed(added_inertia_inv1).cpu().numpy()
        self._B_residual[:, 6:9, 9:12] = torch.diag_embed(added_inertia_inv2).cpu().numpy()
        self._B_residual[:, 9:12, 0:3] = torch.diag_embed(added_mass_inv1).cpu().numpy()
        self._B_residual[:, 9:12, 3:6] = torch.diag_embed(added_mass_inv2).cpu().numpy()

        # run mpc controller
        self._run_mpc()

        # run low level control with updated GRFM
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids) # type: ignore
        self.visualize_marker()
        
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
                self._accel_gyro_mpc, #62:68
                self._gait_contact, #68:70
                self._swing_phase, #70:72
                self._previous_actions, #44:62
            ),
            dim=-1,
        )
        buffer_mask = self.history_buffer.size >= self.num_history
        if buffer_mask.any():
            reset_id = torch.nonzero(buffer_mask, as_tuple=True)[0]
            self.history_buffer.pop(reset_id)
        self.history_buffer.push(self._obs)
        # observation = {"policy": self._obs}
        observation = {"policy": self.history_buffer.data_flat}
        return observation
    
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
    

class HierarchicalArchAccelPF(HierarchicalArch):
    """Hierarchical Architecture with linear/angular acceleration and sagital foot placement.
    """
    cfg: HierarchicalArchAccelPFCfg
    
    def __init__(self, cfg: HierarchicalArchCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
    def _split_action(self, policy_action:torch.Tensor)->tuple:
        """
        Split policy action into centroidal acceleration and foot height.
        """
        centroidal_acceleration = policy_action[:, :3]
        centroidal_ang_acceleration = policy_action[:, 3:6]
        residual_foot_placement = policy_action[:, 6:10]
        return centroidal_acceleration, centroidal_ang_acceleration, residual_foot_placement
    
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
        # process rl actions
        centroidal_accel, centroidal_ang_accel, residual_foot_placement = self._split_action(self._actions_op)
        saggital_residual_foot_placement = residual_foot_placement[:, 0:2] # left, right sagittal
        lateral_residual_foot_placement = residual_foot_placement[:, 2:] # left, right lateral
        
        # transform from local to global frame
        centroidal_accel = torch.bmm(self._root_rot_mat, centroidal_accel.unsqueeze(-1)).squeeze(-1)
        centroidal_ang_accel = torch.bmm(self._root_rot_mat, centroidal_ang_accel.unsqueeze(-1)).squeeze(-1)
        self._A_residual[:, 6:9, -1] = centroidal_accel.cpu().numpy()
        self._A_residual[:, 9:12, -1] = centroidal_ang_accel.cpu().numpy()
        self._foot_placement_residuals[:, 0] = (saggital_residual_foot_placement[:, 0] * torch.cos(self._root_yaw.squeeze()) - lateral_residual_foot_placement[:, 0] * torch.sin(self._root_yaw.squeeze())).cpu().numpy()
        self._foot_placement_residuals[:, 1] = (saggital_residual_foot_placement[:, 0] * torch.sin(self._root_yaw.squeeze()) + lateral_residual_foot_placement[:, 0] * torch.cos(self._root_yaw.squeeze())).cpu().numpy()
        self._foot_placement_residuals[:, 2] = (saggital_residual_foot_placement[:, 1] * torch.cos(self._root_yaw.squeeze()) - lateral_residual_foot_placement[:, 1] * torch.sin(self._root_yaw.squeeze())).cpu().numpy()
        self._foot_placement_residuals[:, 3] = (saggital_residual_foot_placement[:, 1] * torch.sin(self._root_yaw.squeeze()) + lateral_residual_foot_placement[:, 1] * torch.cos(self._root_yaw.squeeze())).cpu().numpy()
        
        # run mpc controller
        self._run_mpc()
        
        # run low level control with updated GRFM
        joint_torque_augmented = np.zeros((self.num_envs, 10), dtype=np.float32)
        for i in range(len(self.mpc)):
            joint_torque_augmented[i] = self.mpc[i].get_action()
        self._joint_actions = torch.from_numpy(joint_torque_augmented).to(self.device).view(self.num_envs, -1)
        self._robot_api.set_joint_effort_target(self._joint_actions, self._joint_ids)
        
        self.visualize_marker()
    
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
                self._previous_actions, #44:54
                self._accel_gyro_mpc, #54:60
                self._gait_contact, #60:62
                self._swing_phase, #62:64
            ),
            dim=-1,
        )
        observation = {"policy": self._obs}
        return observation
    
    def log_action(self)->None:
        log = {}
        if self.common_step_counter % self.cfg.num_steps_per_env:
            centroidal_acceleration = self._actions[0, :3].cpu().numpy()
            centroidal_ang_acceleration = self._actions[0, 3:6].cpu().numpy()
            foot_placement = self._actions[0, 6:].cpu().numpy()
            log["raw_action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["raw_action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["raw_action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["raw_action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["raw_action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["raw_action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
            log["raw_action/sagital_foot_placement_left"] = foot_placement[0]
            log["raw_action/lateral_foot_placement_left"] = foot_placement[1]
            log["raw_action/sagital_foot_placement_right"] = foot_placement[2]
            log["raw_action/lateral_foot_placement_right"] = foot_placement[3]
            
            centroidal_acceleration = self._actions_op[0, :3].cpu().numpy()
            centroidal_ang_acceleration = self._actions_op[0, 3:6].cpu().numpy()
            foot_placement = self._actions[0, 6:].cpu().numpy()
            log["action/centroidal_acceleration_x"] = centroidal_acceleration[0]
            log["action/centroidal_acceleration_y"] = centroidal_acceleration[1]
            log["action/centroidal_acceleration_z"] = centroidal_acceleration[2]
            log["action/centroidal_ang_acceleration_x"] = centroidal_ang_acceleration[0]
            log["action/centroidal_ang_acceleration_y"] = centroidal_ang_acceleration[1]
            log["action/centroidal_ang_acceleration_z"] = centroidal_ang_acceleration[2]
            log["raw_action/sagital_foot_placement_left"] = foot_placement[0]
            log["raw_action/lateral_foot_placement_left"] = foot_placement[1]
            log["raw_action/sagital_foot_placement_right"] = foot_placement[2]
            log["raw_action/lateral_foot_placement_right"] = foot_placement[3]
            
        self.extras["log"].update(log)