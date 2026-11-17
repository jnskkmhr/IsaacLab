# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position actions with an explicit policy-controlled subset."""

from isaaclab.utils import configclass

from isaaclab_contrib.mdp import MirrorJointPositionActionCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints

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
    """The student controls legs, waist, and both arms in that order."""

    joint_pos = MirrorJointPositionActionCfg(
        asset_name="robot",
        joint_names=ALL_JOINTS,
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": ALL_JOINTS},
    )
