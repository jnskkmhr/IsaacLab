# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_deploy.mdp as g1_mdp

@configclass
class G1RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=vel_mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)} 
    )
    track_ang_vel_z = RewTerm(
        func=vel_mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)} 
    )
    
    # -- general style penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0) 
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01) 

    # -- base penalties
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0) 
    # body_orientation_l2 = RewTerm(
    #     func=vel_mdp.body_orientation_l2, 
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, 
    #     weight=-2.0
    # )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0) 
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # TODO: better to have angular momentum penalty here

    # -- joint penalties 
    energy = RewTerm(func=g1_mdp.energy, weight=-1e-3)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    
    # penalize joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # regularize leg dofs
    # joint_deviation_legs = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.02,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])},
    # )
    # penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*waist.*")},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll.*", 
                    ".*_shoulder_pitch.*", 
                    ".*_shoulder_yaw.*", 
                    ".*_elbow.*", 
                ],
            )
        },
    )
    joint_deviation_wrist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_wrist.*", 
                ],
            )
        },
    )

    # # -- gait
    # feet_swing = RewTerm(
    #     func=g1_mdp.reward_feet_swing,
    #     weight=2.0,
    #     params={
    #         "swing_period": 0.4 * 2,
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=".*_ankle_roll_link", 
    #         ),
    #     },
    # )

    gait = RewTerm(
        func=g1_mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.4*2,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # feet_air_time = RewTerm(
    #     func=vel_mdp.feet_air_time_positive_biped,
    #     weight=0.75,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.4,
    #     },
    # )

    # # soft contact
    # feet_air_time_soft_contact = RewTerm(
    #     func=g1_mdp.feet_air_time_positive_biped,
    #     weight=0.15,
    #     params={
    #         "command_name": "base_velocity",
    #         "action_term_name": "physics_callback",
    #         "threshold": 0.4,
    #     },
    # )

    # -- stance foot
    # penalize lateral foot distance
    foot_distance = RewTerm(
        func = g1_mdp.reward_foot_distance,
        weight=-1.0,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*_ankle_roll_link",
                preserve_order=True,
            ),
        },
    )

    # feet_stumble = RewTerm(
    #     func=g1_mdp.feet_stumble,
    #     weight=-2.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll.*"])},
    # )

    feet_slide = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # # soft contact
    # feet_slide_soft_contact = RewTerm(
    #     func=g1_mdp.feet_slide,
    #     weight=-0.25,
    #     params={
    #         "action_term_name": "physics_callback",
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     },
    # )

    
    # -- swing foot rewards

    # encourage specific foot clearance value 
    foot_clearance = RewTerm(
        func=g1_mdp.foot_clearance_reward,
        weight=2.0,
        params={
            "target_height": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )