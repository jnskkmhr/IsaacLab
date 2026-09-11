# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actions for the G1 29-DoF granular locomotion task.

Identical to the soft-contact task except that the analytical contact-solver action is dropped:
here ground reaction is produced by the MPM solver through the coupled foot proxies.
"""

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

FOOT_SOLE_EDGE_X = (-0.065, 0.141)
"""Sole extent along the foot's forward axis relative to the ankle roll link [m]."""

FOOT_SOLE_EDGE_Y = (-0.0368, 0.0368)
"""Sole extent along the foot's lateral axis relative to the ankle roll link [m]."""

FOOT_SOLE_Z = -0.03539
"""Sole height relative to the ankle roll link origin [m]."""

ACTIVE_JOINT = [
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


@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIVE_JOINT,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
