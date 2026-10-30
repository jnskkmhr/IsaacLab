# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_assets import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE

from isaaclab_tasks.manager_based.soft_contact import (
    BoxColliderCfg,
    PhysicsCallbackActionCfg,
)

collider_cfg = BoxColliderCfg(
    contact_edge_x=(-0.065, 0.141),
    contact_edge_y=(-0.0368, 0.0368),
    contact_edge_z=(-0.03539, 0.0),
    resolution=(5, 5),
)

SOFT_CONTACT_THRESHOLD = 40.0

@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""
    
    joint_pos = mdp.JointPositionActionCfg(
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
        # scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE, 
        scale=1.0, # TODO: experiment with this scale
        use_default_offset=True, 
        preserve_order=True,
        )
    
    # 3D RFT soft contact
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        backend="3D-warp",
        # backend="2D-warp",
        intruder_geometry_cfg=collider_cfg,
        enable_ema_filter=True,
        contact_threshold=SOFT_CONTACT_THRESHOLD,
        contact_data_history_length=10,
        history_logging_decimation=10,
        contact_vis_force_threshold=SOFT_CONTACT_THRESHOLD,
    )