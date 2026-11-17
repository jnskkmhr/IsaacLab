# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position actions with an explicit policy-controlled subset."""

from isaaclab.utils import configclass

from isaaclab_contrib.mdp import MirrorJointPositionActionCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints

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

WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]


@configclass
class ActionsCfg:
    """Only joint_pos contributes policy actions."""

    joint_pos = MirrorJointPositionActionCfg(
        asset_name="robot",
        joint_names=UPPER_JOINTS,
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": UPPER_JOINTS},
    )
