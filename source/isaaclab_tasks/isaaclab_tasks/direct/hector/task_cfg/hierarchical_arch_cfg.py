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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Task core
from isaaclab_tasks.direct.hector.common.task_reward import VelocityTrackingReward, AliveReward, ContactTrackingReward, PoseTrackingReward
from isaaclab_tasks.direct.hector.common.task_penalty import VelocityTrackingPenalty, TwistPenalty, FeetSlidePenalty, JointPenalty, ActionSaturationPenalty

from isaaclab_tasks.direct.hector.common.sampler import UniformLineSampler, UniformCubicSampler, GridCubicSampler, QuaternionSampler, CircularSampler
from isaaclab_tasks.direct.hector.common.curriculum import CurriculumRateSampler, CurriculumLineSampler, CurriculumUniformLineSampler, CurriculumUniformCubicSampler, CurriculumQuaternionSampler
from isaaclab_tasks.direct.hector.core_cfg.terrain_cfg import CurriculumFrictionPatchTerrain

# Task cfg
from isaaclab_tasks.direct.hector.task_cfg.base_arch_cfg import BaseArchCfg

# macros 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
ENV_REGEX_NS = "/World/envs/env_.*"


@configclass
class HierarchicalArchCfg(BaseArchCfg):
    # ================================
    # Common configurations
    # ================================
    seed = 10
    
    reference_height = 0.51
    nominal_gait_stepping_frequency = 1.0
    nominal_foot_height = 0.12
    
    terrain = CurriculumFrictionPatchTerrain
    
    # RL observation, action parameters
    action_space = 3
    observation_space = 54
    state_space = 54
    num_joint_actions = 10 # joint torque
    num_states = 33 # mpc state numbers
    
    # action space
    action_lb = [-1.0,-1.0,-6.0]
    action_ub = [1.0,  1.0, 6.0]
    observation_lb = -50.0
    observation_ub = 50.0
    clip_action = True # clip to -1 to 1
    scale_action = True # scale max value to action_ub
    
    # scene 
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, 
        env_spacing=0.0,
        )
    
    # Robot pose sampler
    center = (terrain.terrain_generator.num_cols*terrain.terrain_generator.size[0]/2, terrain.terrain_generator.num_rows*terrain.terrain_generator.size[1]/2, 0.0)
    robot_position_sampler = CircularSampler(radius=2.0, z_range=(0.55, 0.55))
    robot_quat_sampler = CurriculumQuaternionSampler(
        x_range_start=(0.0, math.pi/3), x_range_end=(0.0, 2*math.pi),
        rate_sampler=CurriculumRateSampler(function="linear", start=0, end=24*10000)
    )
    
    # Curriculum sampler
    curriculum_max_steps = 10000
    is_inference = True
    if is_inference:
        # inference
        terrain_curriculum_sampler = CurriculumLineSampler(
            x_start=terrain.num_curriculums-1, x_end=terrain.num_curriculums,
            rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
        )
        robot_target_velocity_sampler = CurriculumUniformCubicSampler(
            x_range_start=(0.3, 0.3), x_range_end=(0.3, 0.3), 
            y_range_start=(0.0, 0.0), y_range_end=(0.0, 0.0),
            z_range_start=(0.0, 0.0), z_range_end=(0.0, 0.0),
            # z_range_start=(0.5, 0.5), z_range_end=(0.5, 0.5),
            rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
        )
    else:
        # train
        terrain_curriculum_sampler = CurriculumLineSampler(
            x_start=0, x_end=terrain.num_curriculums-1,
            rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
        )
        robot_target_velocity_sampler = CurriculumUniformCubicSampler(
            x_range_start=(0.1, 0.3), x_range_end=(0.3, 0.5),
            y_range_start=(0.0, 0.0), y_range_end=(0.0, 0.0),
            z_range_start=(0.0, 0.0), z_range_end=(-0.5, 0.5),
            rate_sampler=CurriculumRateSampler(function="linear", start=0, end=curriculum_max_steps)
        )
    
    # reward parameters
    reward_parameter: VelocityTrackingReward = VelocityTrackingReward(height_similarity_weight=0.66, 
                                                            lin_vel_similarity_weight=0.66,
                                                            ang_vel_similarity_weight=0.66,
                                                            height_similarity_coeff=0.5, 
                                                            lin_vel_similarity_coeff=0.5,
                                                            ang_vel_similarity_coeff=0.5,
                                                            height_reward_mode="exponential",
                                                            lin_vel_reward_mode="exponential",
                                                            ang_vel_reward_mode="exponential")
    pose_tracking_reward_parameter: PoseTrackingReward = PoseTrackingReward(
        position_weight=0.66, yaw_weight=0.66, 
        position_coeff=0.8, yaw_coeff=0.8,
        position_reward_mode="exponential", yaw_reward_mode="exponential")
    alive_reward_parameter: AliveReward = AliveReward(alive_weight=0.33)
    contact_tracking_reward_parameter: ContactTrackingReward = ContactTrackingReward(contact_similarity_weight=0.0, contact_reward_mode="square")
    
    # penalty parameters
    penalty_parameter: VelocityTrackingPenalty = VelocityTrackingPenalty(roll_deviation_weight=0.66, 
                                                               pitch_deviation_weight=0.66, 
                                                               action_penalty_weight=0.05, 
                                                               energy_penalty_weight=0.05,
                                                               foot_energy_penalty_weight=0.00)
    twist_penalty_parameter: TwistPenalty = TwistPenalty(vx_bound=(0.25, 0.5), 
                                                         vy_bound=(0.1, 0.5), 
                                                         wz_bound=(0.15, 0.5), 
                                                         vx_penalty_weight=0.1,
                                                         vy_penalty_weight=0.1,
                                                         wz_penalty_weight=0.1)
    action_saturation_penalty_parameter: ActionSaturationPenalty = ActionSaturationPenalty(action_penalty_weight=0.66, action_bound=(0.9, 1.0))
    foot_slide_penalty_parameter: FeetSlidePenalty = FeetSlidePenalty(feet_slide_weight=0.1)
    
    # Do not use for now
    left_hip_roll_joint_penalty_parameter: JointPenalty = JointPenalty(joint_penalty_weight=0.0, joint_pos_bound=(torch.pi/10, torch.pi/6))
    right_hip_roll_joint_penalty_parameter: JointPenalty = JointPenalty(joint_penalty_weight=0.0, joint_pos_bound=(torch.pi/10, torch.pi/6))
    hip_pitch_joint_deviation_penalty_parameter: JointPenalty = JointPenalty(joint_penalty_weight=0.0, joint_pos_bound=(torch.pi/6, torch.pi/2))


@configclass
class HierarchicalArchPrimeCfg(HierarchicalArchCfg):
    episode_length_s =20
    observation_space = 60
    state_space = 60
    action_space = 6
    
    # action bound hyper parameter
    action_lb = [-6.0, -6.0, -6.0] + [-2.0, -2.0, -2.0]
    action_ub = [6.0,  6.0, 6.0] + [2.0, 2.0, 2.0]

@configclass
class HierarchicalArchAccelPFCfg(HierarchicalArchCfg):
    episode_length_s =20
    observation_space = 60
    state_space = 60
    action_space = 8
    
    # action bound hyper parameter
    action_lb = [-3.0, -3.0, -6.0] + [-2.0, -2.0, -2.0] + [-0.02, -0.02]
    action_ub = [3.0,  3.0, 6.0] + [2.0, 2.0, 2.0] + [0.1, 0.1]