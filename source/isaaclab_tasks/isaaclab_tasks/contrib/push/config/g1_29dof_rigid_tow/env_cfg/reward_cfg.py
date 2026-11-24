# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM reward weights and formulas, adapted to flat rigid/RFT contact at world z=0."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.push.config.g1_29dof_mpm_tow.mdp as tow_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact import mdp as soft_mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact.mdp.mpm_align import rewards as local_mdp
from isaaclab_tasks.contrib.velocity.config.vel_mdp.rewards import foot_clearance_reward

FORWARD_LEAN_LIMIT = math.radians(20.0)
"""Forward base pitch that the orientation penalty allows [rad]; the sand tow task's value."""


@configclass
class G1RewardsCfg:
    """Reward terms for the MDP."""

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
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
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
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.75})
    # As in the sand tow task: forward lean into the rope is free up to the limit.
    flat_orientation_l2 = RewTerm(
        func=tow_mdp.lean_orientation_l2, weight=-10.0, params={"forward_lean_limit": FORWARD_LEAN_LIMIT}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.25)
    energy = RewTerm(func=g1_mdp.energy, weight=-0.001)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_deviation = RewTerm(
        func=g1_mdp.variable_posture_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_velocity",
            "weight_standing": {".*": 0.5},
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
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.25,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_yaw.*": 0.35,
                ".*wrist.*": 0.5,
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
    feet_roll = RewTerm(
        func=g1_mdp.reward_feet_roll,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
    )
    feet_roll_diff = RewTerm(
        func=g1_mdp.reward_feet_roll_diff,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
    )
    stance_foot_angle_penalty = RewTerm(
        func=soft_mdp.stance_foot_angle_penalty_hybrid,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
            "height_sensor_cfg": None,
            "angle_tolerance": math.radians(0.0),
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
        },
    )
    feet_air_time = RewTerm(
        func=local_mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "velocity_threshold": 0.05,
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
        },
    )
    no_fly = RewTerm(
        func=local_mdp.no_fly,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "velocity_threshold": 1.0,
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
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
    feet_slide = RewTerm(
        func=local_mdp.feet_slide,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True),
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
        },
    )
    contact_impulse = RewTerm(
        func=soft_mdp.reward_soft_landing_hybrid,
        weight=-0.005,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
        },
    )
    foot_clearance = RewTerm(
        func=foot_clearance_reward,
        weight=5.0,
        params={
            "target_height": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "standing_position_foot_z": 0.03539,
            "height_sensor_cfg": None,
        },
    )
