# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM joint-position actions plus analytical 3D RFT forces on the feet and on the sled."""

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import MirrorActionTermCfg, MirrorJointPositionActionCfg, mirror_identity

from isaaclab_tasks.contrib.soft_contact import BoxColliderCfg, PhysicsCallbackActionCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry

from ..mdp import TetherHoldActionCfg
from .scene_cfg import SLED_ASSET, SLED_SIZE

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
class MirroredPhysicsCallbackActionCfg(PhysicsCallbackActionCfg, MirrorActionTermCfg):
    """The RFT callback has zero policy-action dimensions; its empty action is invariant."""

    mirror = mirror_identity


@configclass
class MirroredTetherHoldActionCfg(TetherHoldActionCfg, MirrorActionTermCfg):
    """The rope term has zero policy-action dimensions; its empty action is invariant."""

    mirror = mirror_identity


@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = MirrorJointPositionActionCfg(
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        asset_name="robot",
        joint_names=ACTIVE_JOINT,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    physics_callback = MirroredPhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        backend="3D-warp",
        enable_ema_filter=False,
        contact_threshold=5.0,
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
    # The sled is a second intruder in the same analytical sand; the term adds no policy action.
    sled_physics_callback = MirroredPhysicsCallbackActionCfg(
        asset_name=SLED_ASSET,
        body_names=[".*"],
        backend="3D-warp",
        enable_ema_filter=False,
        contact_threshold=5.0,
        debug_vis=False,
        intruder_geometry_cfg=BoxColliderCfg(
            contact_edge_x=(-0.5 * SLED_SIZE[0], 0.5 * SLED_SIZE[0]),
            contact_edge_y=(-0.5 * SLED_SIZE[1], 0.5 * SLED_SIZE[1]),
            contact_edge_z=(-0.5 * SLED_SIZE[2], 0.5 * SLED_SIZE[2]),
            resolution=(5, 5),
        ),
    )
    # After both contact terms: they replace the permanent wrenches, including the rope force.
    tether_hold = MirroredTetherHoldActionCfg(asset_name="robot")
