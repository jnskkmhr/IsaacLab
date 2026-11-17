# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as Term
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid import mdp as rigid_mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact import mdp as soft_mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact.mdp.flat_rewards import (
    action_rate_l2 as reference_action_rate_l2,
)


@configclass
class G1RewardsCfg:
    """Reward terms and parameters saved in tyiby5k6."""

    track_lin_vel_xy = Term(
        func=rigid_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4,
        params={"std": 0.7071067811865476, "command_name": "base_velocity"},
    )

    track_heading = Term(
        func=rigid_mdp.track_heading_world_exp,
        weight=4,
        params={
            "std": 0.7071067811865476,
            "asset_cfg": SceneEntityCfg(name="robot", preserve_order=False),
            "command_name": "base_velocity",
        },
    )

    track_ang_vel_z = Term(
        func=rigid_mdp.track_ang_vel_z_world_exp,
        weight=1,
        params={"std": 0.7071067811865476, "command_name": "base_velocity"},
    )

    termination_penalty = Term(func=mdp.is_terminated, weight=-200, params={})

    action_rate_l2_lower_body = Term(
        func=reference_action_rate_l2, weight=-0.01, params={"joint_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
    )

    action_rate_l2_upper_body = Term(
        func=reference_action_rate_l2,
        weight=-0.05,
        params={"joint_idx": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]},
    )

    base_height = Term(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.75})

    flat_orientation_l2 = Term(func=mdp.flat_orientation_l2, weight=-10, params={})

    lin_vel_z_l2 = Term(func=mdp.lin_vel_z_l2, weight=-1, params={})

    ang_vel_xy_l2 = Term(func=mdp.ang_vel_xy_l2, weight=-0.05, params={})

    undesired_contacts = Term(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names="(?!.*ankle.*).*", preserve_order=False),
        },
    )

    energy = Term(func=rigid_mdp.energy, weight=-0.001, params={})

    dof_vel_l2 = Term(func=mdp.joint_vel_l2, weight=-0.0002, params={})

    dof_acc_l2 = Term(func=mdp.joint_acc_l2, weight=-2.5e-07, params={})

    dof_pos_limits = Term(
        func=mdp.joint_pos_limits, weight=-2, params={"asset_cfg": SceneEntityCfg(name="robot", preserve_order=False)}
    )

    joint_deviation = Term(
        func=rigid_mdp.variable_posture_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=".*", preserve_order=False),
            "command_name": "base_velocity",
            "weight_running": {
                ".*knee.*": 0.005,
                ".*elbow.*": 0.25,
                ".*wrist.*": 0.5,
                ".*hip_yaw.*": 0.08,
                ".*hip_roll.*": 0.3,
                ".*hip_pitch.*": 0.005,
                ".*waist_yaw.*": 0.15,
                ".*ankle_roll.*": 0.02,
                ".*waist_roll.*": 2,
                ".*ankle_pitch.*": 0.02,
                ".*waist_pitch.*": 2,
                ".*shoulder_yaw.*": 0.35,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_pitch.*": 0.5,
            },
            "weight_walking": {
                ".*knee.*": 0.02,
                ".*elbow.*": 0.25,
                ".*wrist.*": 0.5,
                ".*hip_yaw.*": 0.08,
                ".*hip_roll.*": 0.3,
                ".*hip_pitch.*": 0.02,
                ".*waist_yaw.*": 0.15,
                ".*ankle_roll.*": 0.02,
                ".*waist_roll.*": 2,
                ".*ankle_pitch.*": 0.02,
                ".*waist_pitch.*": 2,
                ".*shoulder_yaw.*": 0.35,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_pitch.*": 0.5,
            },
            "weight_standing": {".*": 0.5},
            "running_threshold": 2,
            "walking_threshold": 0.05,
        },
    )

    feet_roll = Term(
        func=rigid_mdp.reward_feet_roll,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
    )

    feet_roll_diff = Term(
        func=rigid_mdp.reward_feet_roll_diff,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
    )

    stance_foot_angle_penalty = Term(
        func=soft_mdp.stance_foot_angle_penalty_hybrid,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*ankle_roll.*"], preserve_order=True
            ),
            "soft_contact_sensor_name": "physics_callback",
            "height_sensor_cfg": None,  # Flat terrain: the surface normal is vertical.
            "angle_tolerance": 0.0,
        },
    )

    feet_air_time = Term(
        func=soft_mdp.feet_air_time_positive_biped_hybrid,
        weight=2,
        params={
            "threshold": 0.5,
            "command_name": "base_velocity",
            "velocity_threshold": 0.05,
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                name="contact_forces", body_names=".*ankle_roll.*", preserve_order=False
            ),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

    no_fly = Term(
        func=soft_mdp.no_fly_hybrid,
        weight=-2,
        params={
            "command_name": "base_velocity",
            "velocity_threshold": 1.5,
            "soft_contact_threshold": 40,
            "rigid_contact_threshold": 5,
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                name="contact_forces", body_names=".*ankle_roll.*", preserve_order=False
            ),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

    foot_distance = Term(
        func=rigid_mdp.reward_foot_lateral_symmetry,
        weight=-2,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*_ankle_roll_link", preserve_order=True),
        },
    )

    feet_slide = Term(
        func=soft_mdp.feet_slide_hybrid,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*ankle_roll.*", preserve_order=False),
            "soft_contact_threshold": 40,
            "rigid_contact_threshold": 5,
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                name="contact_forces", body_names=".*ankle_roll.*", preserve_order=True
            ),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

    contact_impulse = Term(
        func=soft_mdp.reward_soft_landing_hybrid,
        weight=-0.005,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                name="contact_forces", body_names=".*ankle_roll.*", preserve_order=False
            ),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

    foot_clearance = Term(
        func=rigid_mdp.foot_clearance_reward,
        weight=5,
        params={
            "std": 0.05,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*_ankle_roll_link", preserve_order=False),
            "tanh_mult": 2,
            "target_height": 0.1,
            "standing_position_foot_z": 0.03539,
        },
    )
