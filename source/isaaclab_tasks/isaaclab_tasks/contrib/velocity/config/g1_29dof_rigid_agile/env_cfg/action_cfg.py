# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import MirrorJointPositionActionCfg

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp as g1_mdp

from ..mdp import symmetry

CONTROLLED_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # "left_shoulder_pitch_joint",
    # "left_shoulder_roll_joint",
    # "left_shoulder_yaw_joint",
    # "left_elbow_joint",
    # "left_wrist_roll_joint",
    # "left_wrist_pitch_joint",
    # "left_wrist_yaw_joint",
    # "right_shoulder_pitch_joint",
    # "right_shoulder_roll_joint",
    # "right_shoulder_yaw_joint",
    # "right_elbow_joint",
    # "right_wrist_roll_joint",
    # "right_wrist_pitch_joint",
    # "right_wrist_yaw_joint",
]


@configclass
class G1ActionsCfg:
    """Ordered leg and waist joint-position targets, including waist yaw."""

    joint_pos = MirrorJointPositionActionCfg(
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": CONTROLLED_JOINT_NAMES},
        asset_name="robot",
        joint_names=CONTROLLED_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-10.0, 10.0)},
    )

    random_pos = g1_mdp.RandomActionCfg(
        asset_name="robot",
        joint_names_exclude=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"],
        sample_range=(0.1, 2.5),
        preserve_order=True,
        velocity_profile_cfg=g1_mdp.TrapezoidalVelocityProfileCfg(
            acceleration_range=(1.0, 20.0),
            max_velocity_range=(10.0, 20.0),
            min_cruise_ratio=0.1,
            synchronize_joints=True,
            time_scaling_method="max_time",
            use_smooth_start=False,
            position_tolerance=0.001,
            velocity_tolerance=0.01,
            enable_position_limits=True,
            enable_velocity_limits=True,
        ),
    )
