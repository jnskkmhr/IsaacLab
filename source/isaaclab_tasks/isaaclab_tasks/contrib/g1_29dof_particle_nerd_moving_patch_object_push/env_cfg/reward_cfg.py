# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object direction progress with contact, posture and locomotion regularization."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp

from .. import mdp as push_mdp
from .action_cfg import ALL_JOINTS


@configclass
class RewardsCfg:
    object_progress = RewTerm(
        func=push_mdp.direction_progress, weight=8.0, params={"command_name": "object_direction", "speed_scale": 0.8}
    )
    object_turn = RewTerm(
        func=push_mdp.object_turning, weight=2.0, params={"command_name": "object_direction", "std": 0.5}
    )
    hand_contact = RewTerm(
        func=push_mdp.hand_contact,
        weight=1.0,
        params={
            "command_name": "object_direction",
            "asset_cfg": (
                SceneEntityCfg("robot", body_names="left_(wrist_yaw|hand_.*)_link"),
                SceneEntityCfg("robot", body_names="right_(wrist_yaw|hand_.*)_link"),
            ),
            "force_scale": 1.0,
        },
    )
    hand_tracking = RewTerm(
        func=push_mdp.hand_tracking, weight=1.0, params={"command_name": "object_direction", "std": 0.2}
    )
    stand_still = RewTerm(
        func=push_mdp.stand_still,
        weight=5.0,
        params={"command_name": "object_direction", "position_std": 0.15, "velocity_std": 0.2},
    )
    no_touch_when_standing = RewTerm(
        func=push_mdp.no_touch_when_standing, weight=-1.0, params={"command_name": "object_direction"}
    )
    posture = RewTerm(
        func=push_mdp.posture_tracking,
        weight=5.0,
        params={"command_name": "object_direction", "height_std": 0.15, "pitch_std": 0.25},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)
    waist_velocity = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")}
    )
    leg_posture = RewTerm(
        func=push_mdp.leg_posture_when_standing,
        weight=-1.0,
        params={
            "command_name": "object_direction",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
        },
    )
    feet_air_time = RewTerm(
        func=push_mdp.feet_air_time, weight=1.0, params={"command_name": "object_direction", "threshold": 0.5}
    )
    feet_slide = RewTerm(
        func=push_mdp.feet_slide,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link", preserve_order=True)},
    )
    foot_distance = RewTerm(
        func=push_mdp.minimum_foot_separation,
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
    contact_impulse = RewTerm(func=push_mdp.landing_force, weight=-0.005, params={"command_name": "object_direction"})
    stance_foot_angle = RewTerm(
        func=push_mdp.stance_foot_angle_penalty,
        weight=-8.0,
        params={
            "angle_tolerance": 0.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True),
        },
    )
