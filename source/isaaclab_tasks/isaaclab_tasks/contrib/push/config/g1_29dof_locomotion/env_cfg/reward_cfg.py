# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking and lower-body regularization."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as vel_mdp

from .. import mdp
from .action_cfg import LOWER_JOINTS


@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=vel_mdp.track_lin_vel_xy_yaw_frame_exp, weight=4.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z = RewTerm(
        func=vel_mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    base_height = RewTerm(
        func=mdp.commanded_base_height_exp,
        weight=10.0,
        params={"command_name": "base_height", "std": 0.2, "asset_cfg": SceneEntityCfg("robot")},
    )
    trunk_orientation = RewTerm(
        func=mdp.body_flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    # lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    lin_vel_z = RewTerm(
        func=mdp.body_lin_vel_z_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    ang_vel_xy = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    # ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINTS)}
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINTS)}
    )
    # energy = RewTerm(func=g1_mdp.energy, weight=-1e-3)
    waist_posture = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")}
    )
    feet_air_time = RewTerm(
        func=vel_mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
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
        func=g1_mdp.reward_foot_lateral_symmetry,
        weight=-2.0,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True),
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
            "command_name": "base_velocity",
            "command_threshold": 0.05,
        },
    )
    # no_fly = RewTerm(
    #     func=g1_mdp.fly,
    #     weight=-2.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #         "threshold": 5.0,
    #         "command_name": "base_velocity",
    #         "velocity_threshold": 1.0,
    #     },
    # )
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
