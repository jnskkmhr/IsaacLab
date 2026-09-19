# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp


@configclass
class G1RewardsCfg:
    """Reward terms for the MDP."""

    """
    task rewards
    """
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_heading = RewTerm(
        func=g1_mdp.track_heading_world_exp,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )

    """
    regularization.
    """
    action_rate_l2_lower_body = RewTerm(
        func=g1_mdp.action_rate_l2,
        weight=-0.01,
        params={"joint_idx": list(range(12))},  # mjc order
    )
    action_rate_l2_upper_body = RewTerm(
        func=g1_mdp.action_rate_l2,
        weight=-0.05,
        params={"joint_idx": list(range(12, 29))},  # mjc order
    )
    energy = RewTerm(func=g1_mdp.energy, weight=-1e-3)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # penalize joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_deviation = RewTerm(
        func=g1_mdp.variable_posture_l1,  # type: ignore
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_velocity",
            "weight_standing": {
                ".*": 0.5,
            },
            "weight_walking": {
                # leg
                ".*hip_pitch.*": 0.02,
                # ".*hip_roll.*": 0.15,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.15,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                # waist
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 2.0,
                # # arms
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.25,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_yaw.*": 0.35,
                ".*wrist.*": 0.5,
            },
            "weight_running": {
                # leg
                ".*hip_pitch.*": 0.005,
                # ".*hip_roll.*": 0.15,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.15,
                ".*knee.*": 0.005,
                ".*ankle_pitch.*": 0.01,
                ".*ankle_roll.*": 0.01,
                # waist
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 2.0,
                # arms
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.25,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_yaw.*": 0.35,
                ".*wrist.*": 0.5,
            },
            "walking_threshold": 0.05,
            "running_threshold": 2.0,
        },
    )

    """
    style rewards
    """

    # -- base penalties
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.75})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # -- foot orientation
    # feet_yaw_diff = RewTerm(
    #     func=mdp.reward_feet_yaw_diff,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=[".*ankle_roll.*"],
    #             preserve_order=True,
    #         ),
    #     },
    # )

    # feet_yaw_mean = RewTerm(
    #     func=mdp.reward_feet_yaw_mean,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=[".*ankle_roll.*"],
    #             preserve_order=True,
    #         ),
    #     },
    # )

    feet_roll = RewTerm(
        func=g1_mdp.reward_feet_roll,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        },
    )

    feet_roll_diff = RewTerm(
        func=g1_mdp.reward_feet_roll_diff,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        },
    )

    feet_pitch = RewTerm(
        func=g1_mdp.reward_feet_pitch,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        },
    )

    feet_pitch_diff = RewTerm(
        func=g1_mdp.reward_feet_pitch_diff,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        },
    )

    feet_pitch_contact = RewTerm(
        func=g1_mdp.reward_feet_pitch_contact,
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*ankle_roll.*"],
                preserve_order=True,
            ),
        },
    )

    """
    gait
    """

    # rewarded when agent walks with single stance gait.
    # this is sparse reward as agent receives reward equivalent to swing time only during single stance mode.
    feet_air_time = RewTerm(
        func=g1_mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 0.5,
            "velocity_threshold": 0.05,
        },
    )

    # no_fly = RewTerm(
    #     func=mdp.fly,
    #     weight=-1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #         "threshold": 5.0,
    #         "command_name": "base_velocity",
    #         "velocity_threshold": 1.0,
    #     },
    # )

    """
    stance foot
    """

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # penalize lateral foot distance
    foot_distance = RewTerm(
        # func=mdp.reward_foot_distance,
        func=g1_mdp.reward_foot_lateral_symmetry,
        weight=-2.0,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*_ankle_roll_link",
                preserve_order=True,
            ),
        },
    )

    # feet_force = RewTerm(
    #     func=mdp.foot_force,
    #     weight=-5e-3,
    #     # weight=-1e-2,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #         # "threshold": 300,
    #         # "max_reward": 300,
    #         "threshold": 250,
    #         "max_reward": 250,
    #     },
    # )

    contact_impulse = RewTerm(
        func=g1_mdp.reward_soft_landing,
        weight=-5e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
            "command_threshold": 0.05,
        },
    )

    """
    swing foot
    """

    # encourage specific foot clearance value
    foot_clearance = RewTerm(
        func=g1_mdp.foot_clearance_reward,
        weight=5.0,
        params={
            "target_height": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "standing_position_foot_z": 0.03539,
        },
    )
