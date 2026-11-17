# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position actions with an explicit policy-controlled subset."""

from isaaclab.utils import configclass

from isaaclab_contrib.mdp import MirrorJointPositionActionCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp.actions import TrapezoidalVelocityProfileCfg

from ..mdp.actions import RandomArmActionCfg

LOWER_JOINTS = [
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
]
UPPER_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ALL_JOINTS = LOWER_JOINTS + UPPER_JOINTS


@configclass
class G1ActionsCfg:
    """Only joint_pos contributes policy actions."""

    joint_pos = MirrorJointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_JOINTS,
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": LOWER_JOINTS},
    )
    random_arms = RandomArmActionCfg(
        asset_name="robot",
        joint_names=UPPER_JOINTS,
        preserve_order=True,
        sample_range=(2.0, 4.0),
        no_random_when_walking=False,
        joint_pos_limits={
            "left_shoulder_pitch_joint": (-0.8, 0.6),
            "right_shoulder_pitch_joint": (-0.8, 0.6),
            "left_shoulder_roll_joint": (0.0, 0.8),
            "right_shoulder_roll_joint": (-0.8, 0.0),
            "left_shoulder_yaw_joint": (-0.5, 0.5),
            "right_shoulder_yaw_joint": (-0.5, 0.5),
            "left_elbow_joint": (0.1, 1.5),
            "right_elbow_joint": (0.1, 1.5),
            "left_wrist_roll_joint": (-0.4, 0.4),
            "right_wrist_roll_joint": (-0.4, 0.4),
            "left_wrist_pitch_joint": (-0.4, 0.4),
            "right_wrist_pitch_joint": (-0.4, 0.4),
            "left_wrist_yaw_joint": (-0.4, 0.4),
            "right_wrist_yaw_joint": (-0.4, 0.4),
        },
        velocity_profile_cfg=TrapezoidalVelocityProfileCfg(
            acceleration_range=(1.0, 3.0),
            max_velocity_range=(0.5, 1.5),
            synchronize_joints=True,
            use_smooth_start=False,
        ),
    )
