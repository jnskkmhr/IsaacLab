# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1.mdp as t1_mdp

@configclass
class T1RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)} # type: ignore
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)} # type: ignore
    )
    
    
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0) # type: ignore
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0) # type: ignore
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) # type: ignore
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,  # type: ignore
        weight=-1.5e-7, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch", ".*_Ankle_.*"])},
        )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, # type: ignore
        weight=-1.25e-7, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch"])},
        ) 
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005) # type: ignore
    
    # hard contact
    feet_air_time_hard_contact = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.4,
        },
    )
    feet_slide_hard_contact = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )
    
    # soft contact
    feet_air_time_soft_contact = RewTerm(
        func=t1_mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "action_term_name": "physics_callback",
            "threshold": 0.4,
        },
    )
    feet_slide_soft_contact = RewTerm(
        func=t1_mdp.feet_slide,
        weight=-0.1,
        params={
            "action_term_name": "physics_callback",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )
    
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, # type: ignore
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])},
    )
    # joint_deviation_ankle_roll = RewTerm(
    #     func=mdp.joint_deviation_l1, # type: ignore
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Roll"])},
    # )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Shoulder_Pitch",
                    ".*_Shoulder_Roll",
                    ".*_Elbow_Pitch",
                    ".*_Elbow_Yaw",
                    ".*_Wrist_Pitch",
                    ".*_Wrist_Yaw",
                    ".*_Hand_Roll",
                ],
            )
        },
    )
    # joint_deviation_fingers = RewTerm(
    #     func=mdp.joint_deviation_l1, # type: ignore
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_five_joint",
    #                 ".*_three_joint",
    #                 ".*_six_joint",
    #                 ".*_four_joint",
    #                 ".*_zero_joint",
    #                 ".*_one_joint",
    #                 ".*_two_joint",
    #             ],
    #         )
    #     },
    # )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="Waist")},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )
    
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0) # type: ignore
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0) # type: ignore
    undesired_contacts = None