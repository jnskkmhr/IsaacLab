# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as Term
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config import vel_mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid import mdp as rigid_mdp

from .. import mdp as local_mdp


@configclass
class G1RewardsCfg:
    track_lin_vel_xy = Term(
        func=rigid_mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"std": 0.7071067811865476, "command_name": "base_velocity"},
        weight=4,
    )

    track_heading = Term(
        func=rigid_mdp.track_heading_world_exp,
        params={
            "std": 0.7071067811865476,
            "asset_cfg": SceneEntityCfg(name="robot", preserve_order=False),
            "command_name": "base_velocity",
        },
        weight=4,
    )

    track_ang_vel_z = Term(
        func=rigid_mdp.track_ang_vel_z_world_exp,
        params={"std": 0.7071067811865476, "command_name": "base_velocity"},
        weight=1,
    )

    termination_penalty = Term(func=mdp.is_terminated, weight=-200)

    action_rate_l2_lower_body = Term(
        func=local_mdp.action_rate_l2, params={"joint_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}, weight=-0.01
    )

    action_rate_l2_upper_body = Term(
        func=local_mdp.action_rate_l2,
        params={"joint_idx": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]},
        weight=-0.05,
    )

    base_height = Term(func=mdp.base_height_l2, params={"target_height": 0.75}, weight=-10)

    flat_orientation_l2 = Term(func=mdp.flat_orientation_l2, weight=-10)

    lin_vel_z_l2 = Term(func=mdp.lin_vel_z_l2, weight=-1)

    ang_vel_xy_l2 = Term(func=mdp.ang_vel_xy_l2, weight=-0.05)

    undesired_contacts = Term(
        func=mdp.undesired_contacts,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names="(?!.*ankle.*).*", preserve_order=False),
        },
        weight=-1,
    )

    energy = Term(func=rigid_mdp.energy, weight=-0.001)

    dof_vel_l2 = Term(func=mdp.joint_vel_l2, weight=-0.0002)

    dof_acc_l2 = Term(func=mdp.joint_acc_l2, weight=-2.5e-07)

    dof_pos_limits = Term(
        func=mdp.joint_pos_limits, params={"asset_cfg": SceneEntityCfg(name="robot", preserve_order=False)}, weight=-2
    )

    joint_deviation = Term(
        func=rigid_mdp.variable_posture_l1,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=".*", preserve_order=False),
            "command_name": "base_velocity",
            "weight_running": {
                ".*knee.*": 0.005,
                ".*elbow.*": 0.25,
                ".*wrist.*": 0.5,
                ".*hip_yaw.*": 0.15,
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
                ".*hip_yaw.*": 0.15,
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
        weight=-1,
    )

    feet_roll = Term(
        func=rigid_mdp.reward_feet_roll,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
        weight=-1,
    )

    feet_roll_diff = Term(
        func=rigid_mdp.reward_feet_roll_diff,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
        weight=-1,
    )

    feet_pitch = Term(
        func=rigid_mdp.reward_feet_pitch,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
        weight=-4,
    )

    feet_pitch_diff = Term(
        func=rigid_mdp.reward_feet_pitch_diff,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
        weight=-4,
    )

    feet_pitch_contact = Term(
        func=local_mdp.reward_feet_pitch_contact,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*ankle_roll.*"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*ankle_roll.*", preserve_order=False),
        },
        weight=-4,
    )

    feet_air_time = Term(
        func=rigid_mdp.feet_air_time_positive_biped,
        params={
            "threshold": 0.5,
            "command_name": "base_velocity",
            "velocity_threshold": 0.05,
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*ankle_roll.*", preserve_order=False),
        },
        weight=0.5,
    )

    no_fly = Term(
        func=local_mdp.no_fly,
        params={
            "command_name": "base_velocity",
            "velocity_threshold": 1,
            "contact_threshold": 5,
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*ankle_roll.*", preserve_order=False),
        },
        weight=-1,
    )

    foot_distance = Term(
        func=rigid_mdp.reward_foot_lateral_symmetry,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*_ankle_roll_link", preserve_order=True),
        },
        weight=-2,
    )

    feet_slide = Term(
        func=local_mdp.feet_slide,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*ankle_roll.*", preserve_order=False),
            "contact_threshold": 5,
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*ankle_roll.*", preserve_order=True),
        },
        weight=-0.25,
    )

    contact_impulse = Term(
        func=rigid_mdp.reward_soft_landing,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*ankle_roll.*", preserve_order=False),
        },
        weight=-0.005,
    )

    foot_clearance = Term(
        func=vel_mdp.foot_clearance_reward,
        params={
            "std": 0.05,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*_ankle_roll_link", preserve_order=False),
            "tanh_mult": 2,
            "target_height": 0.1,
            "standing_position_foot_z": 0.03539,
        },
        weight=5,
    )


from isaaclab_tasks.contrib.velocity.config.vel_mdp import rough_height


@configclass
class G1RoughRewardsCfg(G1RewardsCfg):
    base_height = G1RewardsCfg().base_height.replace(
        params={"target_height": 0.75, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    foot_clearance = G1RewardsCfg().foot_clearance.replace(
        func=rough_height.foot_clearance_reward,
        params={
            **G1RewardsCfg().foot_clearance.params,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "left_sensor_cfg": SceneEntityCfg("left_foot_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_scanner"),
        },
    )
