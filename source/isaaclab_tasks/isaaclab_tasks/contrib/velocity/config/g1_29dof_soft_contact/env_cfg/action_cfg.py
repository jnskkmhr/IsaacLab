# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from isaaclab_contrib.mdp import MirrorActionTermCfg, MirrorJointPositionActionCfg, mirror_identity

from isaaclab_tasks.contrib.soft_contact import BoxColliderCfg, PhysicsCallbackActionCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints

JOINT_NAMES = [
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
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=JOINT_NAMES, scale=0.25, use_default_offset=True, preserve_order=True
    )
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        backend="3D-warp",
        enable_ema_filter=False,
        contact_threshold=40.0,
        contact_data_history_length=10,
        history_logging_decimation=10,
        debug_vis=False,
        intruder_geometry_cfg=BoxColliderCfg(
            contact_edge_x=(-0.065, 0.141),
            contact_edge_y=(-0.0368, 0.0368),
            contact_edge_z=(-0.03539, 0.0),
            resolution=(5, 5),
        ),
    )


@configclass
class MirroredPhysicsCallbackActionCfg(PhysicsCallbackActionCfg, MirrorActionTermCfg):
    """The RFT callback has no policy action channels to reflect."""

    mirror = mirror_identity


@configclass
class G1SymmetricActionsCfg(G1ActionsCfg):
    joint_pos = MirrorJointPositionActionCfg(
        **vars(G1ActionsCfg().joint_pos.copy()), mirror=mirror_g1_joints, mirror_params={"joint_names": JOINT_NAMES}
    )
    physics_callback = MirroredPhysicsCallbackActionCfg(**vars(G1ActionsCfg().physics_callback.copy()))


@configclass
class G1RoughActionsCfg(G1SymmetricActionsCfg):
    physics_callback = G1SymmetricActionsCfg().physics_callback.replace(terrain_name="terrain")
