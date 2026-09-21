# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.action_cfg import CONTROLLED_JOINTS
from isaaclab_tasks.contrib.soft_contact import BoxColliderCfg, PhysicsCallbackActionCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints
from isaaclab_tasks.contrib.velocity.config.vel_mdp import (
    MirrorActionTermCfg,
    MirrorJointPositionActionCfg,
    mirror_identity,
)


@configclass
class MirrorPhysicsCallbackActionCfg(PhysicsCallbackActionCfg, MirrorActionTermCfg):
    """Soft-contact callback with an identity mirror for its empty policy action slice."""

    mirror = mirror_identity


@configclass
class G1ActionsCfg:
    """Mimic joint actions and the zero-dimensional soft-contact callback."""

    joint_pos = MirrorJointPositionActionCfg(
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": CONTROLLED_JOINTS},
        asset_name="robot",
        joint_names=CONTROLLED_JOINTS,
        scale=0.2,
        use_default_offset=True,
        preserve_order=True,
    )

    physics_callback = MirrorPhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        backend="3D-warp",
        intruder_geometry_cfg=BoxColliderCfg(
            contact_edge_x=(-0.065, 0.141),
            contact_edge_y=(-0.0368, 0.0368),
            contact_edge_z=(-0.03539, 0.0),
            resolution=(5, 5),
        ),
        enable_ema_filter=False,
        contact_threshold=40.0,
        debug_vis=False,
        contact_data_history_length=10,
        history_logging_decimation=10,
        contact_vis_force_threshold=40.0,
    )
