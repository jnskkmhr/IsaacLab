# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Box velocity tracking, hand proximity, and whole-body regularization."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.push.config.g1_29dof_locomotion.mdp as locomotion_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as vel_mdp

from .. import mdp
from .action_cfg import ALL_JOINTS
from .commands_cfg import STANDING_COMMAND_THRESHOLD


@configclass
class RewardsCfg:
    box_lin_vel = RewTerm(
        func=mdp.box_velocity_tracking_with_hand_contact,
        weight=8.0,
        params={
            "command_name": "box_velocity",
            "std": 0.2,
            "hand_distance": 0.2,
            "command_names": ("hand_pose",),
            "sensor_cfg": (
                SceneEntityCfg("box_contacts", body_names="left_(wrist_yaw|hand_.*)_link"),
                SceneEntityCfg("box_contacts", body_names="right_(wrist_yaw|hand_.*)_link"),
            ),
            "force_threshold": 1.0,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
    box_ang_vel = RewTerm(
        func=mdp.box_yaw_velocity_tracking,
        weight=2.0,
        params={
            "command_name": "box_velocity",
            "std": 0.2,
            "sensor_cfg": (
                SceneEntityCfg("box_contacts", body_names="left_(wrist_yaw|hand_.*)_link"),
                SceneEntityCfg("box_contacts", body_names="right_(wrist_yaw|hand_.*)_link"),
            ),
            "force_threshold": 1.0,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
    hand_box_contact = RewTerm(
        func=mdp.hand_box_contact_fraction,
        weight=1.0,
        params={
            "command_name": "box_velocity",
            "sensor_cfg": (
                SceneEntityCfg("box_contacts", body_names="left_(wrist_yaw|hand_.*)_link"),
                SceneEntityCfg("box_contacts", body_names="right_(wrist_yaw|hand_.*)_link"),
            ),
            "force_threshold": 1.0,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
    hand_position = RewTerm(
        func=mdp.hand_position_tracking_when_moving,
        weight=1.0,
        params={
            "command_name": "box_velocity",
            "command_names": ("hand_pose",),
            "std": 0.2,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
    stand_still_velocity = RewTerm(
        func=mdp.stand_still_velocity_exp,
        weight=5.0,
        params={
            "command_name": "box_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "linear_std": 0.2,
            "angular_std": 0.5,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
    stand_still_position = RewTerm(
        func=mdp.stand_still_position_exp,
        weight=5.0,
        params={
            "command_name": "box_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.15,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
    base_height = RewTerm(
        func=locomotion_mdp.commanded_base_height_l2, weight=-10.0, params={"command_name": "base_height"}
    )
    # orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    orientation = RewTerm(
        func=mdp.body_flat_orientation_l2,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    lin_vel_z = RewTerm(
        func=mdp.body_lin_vel_z_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    # ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    ang_vel_xy = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS)}
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS)}
    )
    waist_posture = RewTerm(
        func=mdp.joint_deviation_l1, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")}
    )
    waist_velocity = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")}
    )
    leg_joint_deviation_l1 = RewTerm(
        func=mdp.leg_joint_deviation_l1,
        weight=-1.0,
        params={
            "command_name": "box_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            "command_threshold": 0.05,
        },
    )
    feet_air_time = RewTerm(
        func=vel_mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "box_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
        },
    )
    feet_slide = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    foot_distance = RewTerm(
        func=g1_mdp.feet_min_lateral_distance_l2,
        weight=-2.0,
        params={
            "min_separation": 0.15,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
        },
    )
    feet_yaw_diff = RewTerm(
        func=g1_mdp.reward_feet_yaw_diff,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True)},
    )
    feet_yaw_mean = RewTerm(
        func=g1_mdp.reward_feet_yaw_mean,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True)},
    )
    contact_impulse = RewTerm(
        func=g1_mdp.reward_soft_landing,
        weight=-5e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "box_velocity",
            "command_threshold": 0.05,
        },
    )
    stance_foot_angle_penalty = RewTerm(
        func=g1_mdp.stance_foot_angle_penalty,
        weight=-8.0,
        params={
            "height_sensor_cfg": None,
            "angle_tolerance": 0.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True),
        },
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)
