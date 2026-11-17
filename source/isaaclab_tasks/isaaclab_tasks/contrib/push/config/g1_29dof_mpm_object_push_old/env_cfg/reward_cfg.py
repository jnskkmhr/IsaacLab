# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion and hand/object tracking rewards for whole-body pushing.

A granular bed exerts no force on a rigid contact sensor, so
the gait terms are rebuilt on the contact state that the coupler feeds back from the MPM entry to
the proxy feet. The contact impulse term penalizes touchdown force magnitude using this same
contact state. Foot sliding uses the combined MPM and MJWarp contact state.
``undesired_contacts`` remains disabled because it requires contact data for other bodies.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.mdp as mpm_mdp
import isaaclab_tasks.core.velocity.mdp as mdp

from .. import mdp as push_mdp
from .scene_cfg import SAND_DEPTH


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    """
    task rewards
    """
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_heading = RewTerm(
        func=g1_mdp.track_heading_world_exp,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )

    """
    style rewards
    """
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_rate_l2_lower_body = RewTerm(
        func=g1_mdp.action_rate_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(hip|knee|ankle).*_joint"])},
    )
    action_rate_l2_upper_body = RewTerm(
        func=g1_mdp.action_rate_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(waist|.*_shoulder|.*_elbow|.*_wrist).*_joint"])},
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10,
        params={"target_height": 0.75, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    flat_orientation_l2 = RewTerm(
        func=push_mdp.body_flat_orientation_l2,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    lin_vel_z_l2 = RewTerm(
        func=push_mdp.body_lin_vel_z_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        weight=-0.25,
    )
    ang_vel_xy_l2 = RewTerm(
        func=push_mdp.body_ang_vel_xy_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        weight=-0.25,
    )

    """
    joint regularization.
    """
    energy = RewTerm(func=g1_mdp.energy, weight=-1e-3)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

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
            "weight_standing": {".*(hip|knee|ankle|waist).*": 0.5, ".*(shoulder|elbow|wrist).*": 0.0},
            "weight_walking": {
                ".*hip_pitch.*": 0.02,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.08,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 2.0,
                ".*shoulder_pitch.*": 0.0,
                ".*elbow.*": 0.0,
                ".*shoulder_roll.*": 0.0,
                ".*shoulder_yaw.*": 0.0,
                ".*wrist.*": 0.0,
            },
            "weight_running": {
                ".*hip_pitch.*": 0.005,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.08,
                ".*knee.*": 0.005,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 2.0,
                ".*shoulder_pitch.*": 0.0,
                ".*elbow.*": 0.0,
                ".*shoulder_roll.*": 0.0,
                ".*shoulder_yaw.*": 0.0,
                ".*wrist.*": 0.0,
            },
            "walking_threshold": 0.05,
            "running_threshold": 2.0,
        },
    )

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

    stance_foot_angle_penalty = RewTerm(
        func=mpm_mdp.stance_foot_angle_penalty,  # type: ignore
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "angle_tolerance": math.radians(0.0),
        },
    )

    """
    gait
    """

    feet_air_time = RewTerm(
        func=mpm_mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "velocity_threshold": 0.05,
        },
    )

    no_fly = RewTerm(
        func=mpm_mdp.no_fly,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "velocity_threshold": 1.0,
        },
    )

    """
    Stance foot
    """
    foot_distance = RewTerm(
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

    feet_slide = RewTerm(
        func=mpm_mdp.feet_slide,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True)},
    )

    contact_impulse = RewTerm(
        func=mpm_mdp.reward_soft_landing,
        weight=-5e-3,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
    )

    """
    Swing foot
    """

    foot_clearance = RewTerm(
        func=mpm_mdp.foot_clearance_reward,
        weight=5.0,
        params={
            "target_height": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "standing_position_foot_z": 0.03539,
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "ground_height_offset": SAND_DEPTH,
        },
    )

    hand_position = RewTerm(func=push_mdp.hand_position_tracking, weight=5.0)
    hand_orientation = RewTerm(func=push_mdp.hand_orientation_tracking, weight=1.0)
    object_velocity = RewTerm(func=push_mdp.box_velocity_tracking, weight=3.0)
