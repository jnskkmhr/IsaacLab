# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-gated box progress, a fixed pushing arm posture, and locomotion regularization on rigid ground."""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as vel_mdp
from isaaclab_tasks.contrib.push.config.g1_29dof_object_push_e2e import mdp as push_mdp

from .commands_cfg import (
    BODY_CONTACT_FORCE_SCALE,
    HAND_CONTACT_FORCE_THRESHOLD,
    NON_HAND_BODIES,
    STANDING_COMMAND_THRESHOLD,
    WRIST_LINKS,
)

MAX_FOOT_TRAILING_DISTANCE = 0.25
"""Distance a foot may trail behind the pelvis along the heading without penalty [m].

This is the only setting that differs from ``g1_29dof_object_push_e2e``, which uses 0.30 m.
"""


@configclass
class RewardsCfg:
    """Box progress through wrist contact, plus the rigid-ground locomotion regularizers."""

    """
    task rewards
    """
    object_progress = RewTerm(
        func=push_mdp.object_progress,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "object_name": "box",
            "left_sensor_cfg": SceneEntityCfg("box_contacts", body_names=WRIST_LINKS[0]),
            "right_sensor_cfg": SceneEntityCfg("box_contacts", body_names=WRIST_LINKS[1]),
            "body_sensor_cfg": SceneEntityCfg("box_contacts", body_names=NON_HAND_BODIES),
            "force_threshold": HAND_CONTACT_FORCE_THRESHOLD,
            "body_force_scale": BODY_CONTACT_FORCE_SCALE,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
    hand_contact = RewTerm(
        func=push_mdp.hand_contact,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "left_sensor_cfg": SceneEntityCfg("box_contacts", body_names=WRIST_LINKS[0]),
            "right_sensor_cfg": SceneEntityCfg("box_contacts", body_names=WRIST_LINKS[1]),
            "body_sensor_cfg": SceneEntityCfg("box_contacts", body_names=NON_HAND_BODIES),
            "force_threshold": HAND_CONTACT_FORCE_THRESHOLD,
            "body_force_scale": BODY_CONTACT_FORCE_SCALE,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
    track_lin_vel_xy = RewTerm(
        func=vel_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_heading = RewTerm(
        func=g1_mdp.track_heading_world_exp,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": math.sqrt(0.2)},
    )
    track_ang_vel_z = RewTerm(
        func=vel_mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    """
    contact and object penalties
    """
    non_hand_contact = RewTerm(
        func=push_mdp.non_hand_contact,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("box_contacts", body_names=NON_HAND_BODIES), "force_scale": 50.0},
    )
    object_tilt = RewTerm(func=push_mdp.object_tilt_l2, weight=-10.0, params={"object_name": "box"})

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
        func=push_mdp.base_height_band_l2,
        weight=-50.0,
        # Free band around the standing height; penalize only excursions outside it.
        params={"min_height": 0.65, "max_height": 0.80, "asset_cfg": SceneEntityCfg("robot")},
    )
    # Forward lean up to 0.35 rad is free; roll and backward lean are always penalized.
    torso_lean = RewTerm(
        func=push_mdp.forward_lean_band_l2,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "max_forward_lean": 0.35},
    )
    pelvis_lean = RewTerm(
        func=push_mdp.forward_lean_band_l2,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis"), "max_forward_lean": 0.35},
    )
    # Keep the robot balanced on its own feet instead of planking against the box.
    feet_trailing = RewTerm(
        func=push_mdp.feet_trailing_distance_l2,
        weight=-50.0,
        params={
            "max_trailing_distance": MAX_FOOT_TRAILING_DISTANCE,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
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
            "weight_standing": {".*(hip|knee|ankle|waist).*": 0.5, ".*(shoulder|elbow).*": 0.5, ".*wrist.*": 0.2},
            "weight_walking": {
                ".*hip_pitch.*": 0.02,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.08,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 0.2,
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.5,
                ".*shoulder_roll.*": 0.5,
                ".*shoulder_yaw.*": 0.5,
                ".*wrist.*": 0.2,
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
                ".*waist_pitch.*": 0.2,
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.5,
                ".*shoulder_roll.*": 0.5,
                ".*shoulder_yaw.*": 0.5,
                ".*wrist.*": 0.2,
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
        func=g1_mdp.stance_foot_angle_penalty,
        weight=-8.0,
        params={
            "height_sensor_cfg": None,
            "angle_tolerance": 0.0,
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

    feet_air_time = RewTerm(
        func=g1_mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 0.5,
            "velocity_threshold": 0.05,
        },
    )

    no_fly = RewTerm(
        func=g1_mdp.fly,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 5.0,
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
        func=vel_mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

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
    Swing foot
    """

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
