# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_reach.mdp as reach_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp


@configclass
class G1RewardsCfg:
    """Rewards for reaching and holding the base-pose target."""

    """
    tracking.
    """

    goal_progress = RewTerm(func=reach_mdp.goal_progress, weight=4.0, params={"command_name": "base_pose"})
    goal_position = RewTerm(
        func=reach_mdp.goal_position_exp, weight=4.0, params={"command_name": "base_pose", "std": 1.0}
    )
    goal_position_fine = RewTerm(
        func=reach_mdp.goal_position_exp, weight=4.0, params={"command_name": "base_pose", "std": 0.25}
    )
    goal_heading = RewTerm(
        func=reach_mdp.goal_heading_exp,
        weight=4.0,
        params={"command_name": "base_pose", "std": 0.5, "position_std": 0.75},
    )
    goal_facing = RewTerm(
        func=reach_mdp.goal_facing_exp,
        weight=0.5,
        params={"command_name": "base_pose", "std": 0.5, "distance_threshold": 1.0},
    )
    goal_motion = RewTerm(
        func=reach_mdp.goal_motion_l2, weight=-2.0, params={"command_name": "base_pose", "position_std": 0.35}
    )
    goal_hold = RewTerm(func=reach_mdp.goal_hold, weight=3.0, params={"command_name": "base_pose"})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    """
    joint regularization.
    """

    action_rate_l2_lower_body = RewTerm(func=g1_mdp.action_rate_l2, weight=-0.01, params={"joint_idx": list(range(12))})
    action_rate_l2_upper_body = RewTerm(
        func=g1_mdp.action_rate_l2, weight=-0.05, params={"joint_idx": list(range(12, 29))}
    )
    energy = RewTerm(func=g1_mdp.energy, weight=-0.001)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation = RewTerm(
        func=reach_mdp.variable_posture_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_pose",
            "weight_hold": {
                ".*hip_pitch.*": 0.5,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.5,
                ".*knee.*": 0.1,
                ".*ankle_pitch.*": 0.1,
                ".*ankle_roll.*": 0.1,
                ".*(waist|shoulder|elbow|wrist).*": 0.5,
            },
            "weight_moving": {
                ".*hip_pitch.*": 0.02,
                ".*hip_roll.*": 0.3,
                # ".*hip_yaw.*": 0.08,
                ".*hip_yaw.*": 0.3,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 2.0,
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.25,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_yaw.*": 0.35,
                ".*wrist.*": 0.5,
            },
        },
    )

    """
    base regularization.
    """

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    base_lin_acc_xy_l2 = RewTerm(
        func=reach_mdp.base_acceleration_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot"), "angular": False},
    )
    base_ang_acc_z_l2 = RewTerm(
        func=reach_mdp.base_acceleration_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot"), "angular": True},
    )

    """
    gait regularization.
    """

    # swing_foot_lateral_position = RewTerm(
    #     func=reach_mdp.swing_foot_lateral_position_l2,
    #     weight=-20.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    #         ),
    #         "hip_cfg": SceneEntityCfg(
    #             "robot", body_names=["left_hip_pitch_link", "right_hip_pitch_link"], preserve_order=True
    #         ),
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    #         ),
    #         "lateral_tolerance": 0.05,
    #     },
    # )

    feet_air_time = RewTerm(
        func=reach_mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_pose",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 0.5,
        },
    )
    no_fly = RewTerm(
        func=reach_mdp.fly,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 5.0,
        },
    )
    foot_clearance = RewTerm(
        func=reach_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "command_name": "base_pose",
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "left_sensor_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "reference_foot_height": 0.10,
            "sole_offset": (0.038, 0.0, -0.03539),
            "tanh_mult": 2.0,
        },
    )

    """
    contact regularization.
    """

    contact_impulse = RewTerm(
        func=g1_mdp.reward_soft_landing,
        weight=-0.005,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": None,
        },
    )
    feet_edge = RewTerm(
        func=reach_mdp.feet_edge_penalty,
        weight=-1.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "height_threshold": 0.03,
            "margin": 0.03,
            "contact_threshold": 1.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
