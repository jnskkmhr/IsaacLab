# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp

from .action_cfg import CONTROLLED_JOINT_NAMES

FEET_LINK_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


@configclass
class G1RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    track_lin_vel_xy_exp = RewTerm(
        func=g1_mdp.track_lin_vel_xy_command_width,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.2, "max_std": 0.4, "threshold": 0.5, "full_speed": 1.0},
    )

    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=7.5,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]),
        },
    )

    base_height = RewTerm(
        func=g1_mdp.base_height_exp,
        weight=2.5,
        params={
            "target_height": 0.78,
            "std": 0.1,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    orientation = RewTerm(
        func=g1_mdp.flat_body_orientation_exp,
        weight=5.0,
        params={"std": math.radians(10.0), "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"])},
    )

    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    ankle_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*ankle.*")},
    )

    ankle_roll_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*ankle_roll.*")},
    )

    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    action_rate = RewTerm(
        func=g1_mdp.action_rate_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    action_rate_rate = RewTerm(
        func=g1_mdp.action_rate_rate_l2,
        weight=-0.025,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES), "soft_ratio": 0.9},
    )

    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )

    feet_slip = RewTerm(
        func=g1_mdp.feet_slip,
        weight=-0.05,
        params={
            "contact_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_LINK_NAMES),
            "robot_cfg": SceneEntityCfg("robot", body_names=FEET_LINK_NAMES),
        },
    )

    feet_roll = RewTerm(
        func=g1_mdp.feet_roll_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_LINK_NAMES)},
    )

    feet_yaw_diff = RewTerm(
        func=g1_mdp.feet_yaw_diff_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_LINK_NAMES)},
    )

    feet_yaw_mean = RewTerm(
        func=g1_mdp.feet_yaw_mean_vs_base,
        weight=-2.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=FEET_LINK_NAMES),
            "base_body_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        },
    )

    root_acc = RewTerm(
        func=g1_mdp.body_acc_l2,  # type: ignore
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    feet_distance = RewTerm(
        func=g1_mdp.feet_distance_from_ref,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_LINK_NAMES),
            "ref_distance": 0.2,
        },
    )

    jumping = RewTerm(
        func=g1_mdp.jumping,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_LINK_NAMES),
            "threshold": 10.0,
        },
    )

    feet_signed_clearance = RewTerm(
        func=g1_mdp.feet_signed_clearance,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_LINK_NAMES, preserve_order=True), "minimum": 0.10},
    )
    smooth_observer = RewTerm(
        func=g1_mdp.SmoothContactObserver,
        weight=1.0,
        params={
            "interval_sensor_cfg": SceneEntityCfg(
                "smooth_interval_forces", body_names=FEET_LINK_NAMES, preserve_order=True
            )
        },
    )
    smooth_interval_loading = RewTerm(func=g1_mdp.smooth_contact_cost, weight=-0.5, params={"mode": "interval_loading"})


@configclass
class G1RewardsRoughCfg(G1RewardsCfg):
    """The reference uses the same reward terms on flat and rough terrain."""
