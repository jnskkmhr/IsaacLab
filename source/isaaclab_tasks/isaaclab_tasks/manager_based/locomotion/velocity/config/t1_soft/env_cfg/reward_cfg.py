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
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_soft.mdp as t1_mdp

@configclass
class T1RewardsCfg:
    """Reward terms for the MDP."""

    # -- tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=vel_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=10.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=vel_mdp.track_ang_vel_z_world_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, 
        weight=-100.0,
        params={
            "target_height": 0.68,
        }
    )

    # -- general style penalties
    alive = RewTerm(func=mdp.is_alive, weight=10) 
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-0.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.5) 

    # -- root penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0) # type: ignore
    flat_orientation_l2 = None
    flat_orientation_exp = RewTerm(
        func=t1_mdp.flat_orientation_exp, 
        weight=4.0,
        params={"std": 0.5,
                "asset_cfg": SceneEntityCfg("robot")},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0) 
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0) 
    base_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2, 
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")},
    )

    # -- joint penalties
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-1.25e-7, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip.*", ".*Knee.*"])},
        )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-4, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip.*", ".*Knee.*"])},
        )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_torque_limits = RewTerm(
        func=t1_mdp.joint_torque_limits, 
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",
                ],
                preserve_order=True,
                ),
                "soft_ratio": 1.0},
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        # weight=-0.1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_.*"])},
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, 
        # weight=-0.1,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="Waist")},
    )

    # -- foot orientation penalities
    feet_yaw_diff = RewTerm(
        func=t1_mdp.reward_feet_yaw_diff, 
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        }
    )

    feet_yaw_mean = RewTerm(
        func=t1_mdp.reward_feet_yaw_mean, 
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        }
    )

    feet_roll = RewTerm(
        func=t1_mdp.reward_feet_roll,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    feet_roll_diff = RewTerm(
        func=t1_mdp.reward_feet_roll_diff,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    feet_pitch = RewTerm(
        func=t1_mdp.reward_feet_pitch,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    feet_pitch_diff = RewTerm(
        func=t1_mdp.reward_feet_pitch_diff,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    # -- swing foot penalties
    # Rewards the agent for having feet in the air (e.g., walking, running, not standing still).
    feet_air_time = RewTerm(
        func=vel_mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.34,
        },
    )
    feet_air_time_soft = RewTerm(
        func=t1_mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "action_term_name": "physics_callback",
            "threshold": 0.34,
        },
    )

    # Penalizes the agent for sliding feet while in contact (i.e., wants stepping, not dragging feet)
    feet_slide = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    feet_slide_soft = RewTerm(
        func=t1_mdp.feet_slide,
        weight=-2.0,
        params={
            "action_term_name": "physics_callback",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    feet_swing = RewTerm(
        func=t1_mdp.reward_feet_swing,
        weight=20.0,
        params={
            "swing_period": 0.2,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot_link"
            ),
        },
    )
    feet_swing_soft = RewTerm(
        func=t1_mdp.reward_feet_swing_soft,
        weight=20.0,
        params={
            "swing_period": 0.2,
            "action_term_name": "physics_callback",
        },
    )

    foot_distance = RewTerm(
        func = t1_mdp.reward_foot_distance,
        weight=-1.0,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    foot_clearance = RewTerm(
        func=t1_mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "target_height": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link"),
        },
    )