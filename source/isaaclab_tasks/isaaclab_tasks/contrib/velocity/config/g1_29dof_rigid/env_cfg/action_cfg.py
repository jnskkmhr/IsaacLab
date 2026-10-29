# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import MirrorJointPositionActionCfg, MirrorJointPositionRangeActionCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry
from isaaclab_tasks.utils import PresetCfg


@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = MirrorJointPositionActionCfg(
        mirror=symmetry.mirror_g1_joints,
        asset_name="robot",
        joint_names=[
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
        ],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    joint_pos.mirror_params = {"joint_names": joint_pos.joint_names}


@configclass
class G1JointRangeActionsCfg(G1ActionsCfg):
    """Map normalized actions to symmetric joint ranges around the default pose."""

    joint_pos = MirrorJointPositionRangeActionCfg(
        mirror=symmetry.mirror_g1_joints,
        asset_name="robot",
        joint_names=G1ActionsCfg().joint_pos.joint_names,
        # scale=1.0,
        scale=2.0,
        use_default_offset=True,
        preserve_order=True,
    )
    joint_pos.mirror_params = {"joint_names": joint_pos.joint_names}


@configclass
class G1ActionPresetsCfg(PresetCfg):
    """Select joint-range scaling with ``env.actions=joint_range_action``."""

    constant_scale_action = G1ActionsCfg()
    joint_range_action: G1JointRangeActionsCfg = G1JointRangeActionsCfg()
    # default = constant_scale_action
    default = joint_range_action
