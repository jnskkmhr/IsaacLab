# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_legacy.mdp as t1_mdp

@configclass
class T1RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)} # type: ignore
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)} # type: ignore
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, # type: ignore
        weight=-20.0,
        params={
            "target_height": 0.68,
        }
    )
    
    
    # -- penalties

    ## -- generic
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0) # type: ignore
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005) # type: ignore

    ## -- base penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0) # type: ignore
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2) # type: ignore
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5) # type: ignore

    ## -- joint regularization
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,  # type: ignore
        weight=-1.5e-7, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch", ".*_Ankle_.*"])},
        )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,  # type: ignore
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Waist", ".*_Hip_.*", ".*_Knee_Pitch"])},
        weight=-5e-4, 
        )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, # type: ignore
        weight=-1.25e-7, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch"])},
        ) 

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, # type: ignore
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])},
    )

    #  Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="Waist")},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        # weight=-0.5,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])},
    )
    
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1, # type: ignore
    #     # weight=-0.35,
    #     weight=-0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_Shoulder_Pitch",
    #                 ".*_Shoulder_Roll",
    #                 ".*_Elbow_Pitch",
    #                 ".*_Elbow_Yaw",
    #                 ".*_Wrist_Pitch",
    #                 ".*_Wrist_Yaw",
    #                 ".*_Hand_Roll",
    #             ],
    #         )
    #     },
    # )

    ## -- regulate foot orientation 
    foot_roll = RewTerm(
        func=t1_mdp.reward_feet_roll, # type: ignore
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")},
    )
    foot_pitch = RewTerm(
        func=t1_mdp.reward_feet_pitch, # type: ignore
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")},
    )


    ## -- reward periodic contact sequence 
    foot_contact = RewTerm(
        func=t1_mdp.reward_feet_contact_number,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_foot_link",
                preserve_order=True,
            ),
            "pos_rw": 1.0,
            "neg_rw": -0.3,
        },
    )
    # soft contact version
    foot_contact_soft = RewTerm(
        func=t1_mdp.reward_feet_contact_number_soft,
        weight=2.0,
        params={
            "action_term_name": "physics_callback",
            "pos_rw": 1.0,
            "neg_rw": -0.3,
        },
    )
    
    ## -- reward proper swing foot motion (for hard contact terrain.)
    # Lift your feet off the ground and keep them up for a reasonable amount of time.
    feet_air_time = RewTerm(
        func=t1_mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold_min": 0.05,
            "threshold_max": 0.35,
        },
    )
    ## soft contact version
    feet_air_time_soft_contact = RewTerm(
        func=t1_mdp.feet_air_time_positive_biped_soft,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "action_term_name": "physics_callback",
            "threshold": 0.35,
        },
    )

    # Don't slide when foot is on ground.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # soft contact version
    feet_slide_soft_contact = RewTerm(
        func=t1_mdp.feet_slide_soft,
        weight=-0.25,
        params={
            "action_term_name": "physics_callback",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # Reward foot to track predefined curve
    track_foot_height = RewTerm(
        func=t1_mdp.track_foot_height,
        weight=2.0,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*_foot_link"],
                preserve_order=True,
            ),
        },
    )

    # Guide the foot height during the swing phase.
    # Large penalty when foot is moving fast and far from target height.
    # This is a dense reward.
    foot_clearance = RewTerm(
        func=t1_mdp.feet_clearance,
        weight=-2.0,
        params={
        "target_height": 0.1+0.0305,
        "command_name": "base_velocity",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # # Encourage soft landings.
    # soft_landing = RewTerm(
    #     func=t1_mdp.soft_landing,
    #     weight=-1e-5,
    #     params={
    #     "sensor_name": "contact_forces",
    #     "command_name": "base_velocity",
    #     "command_threshold": 0.05,
    #     },
    # )

    # ensure two feet are separated by a certain distance
    foot_distance = RewTerm(
        func = t1_mdp.reward_foot_distance,
        weight=-0.5,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )