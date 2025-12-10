# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.soft_contact import IntruderGeometryCfg, PhysicsCallbackActionCfg

@configclass    
class G1FootGeometryCfg(IntruderGeometryCfg):
    """Configuration for the intruder geometry used in soft contact modeling."""
    contact_edge_x: tuple[float, float] = (-0.065, 0.141)  # length in x direction (m)
    contact_edge_y: tuple[float, float] = (-0.0368, 0.0368)  # length in y direction (m)
    contact_edge_z: tuple[float, float] = (-0.03539, 0.0)  # length in z direction (m)
    # contact_edge_z: tuple[float, float] = (-0.039, 0.0)  # length in z direction (m)
    num_contact_points: int = 5 * 5
    # num_contact_points: int = 20 * 20 # inference

@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        # joint_names=[".*"], 
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
        scale=0.25, # leggedlab 
        # scale=0.5, # mjlab
        use_default_offset=True, 
        ) 
    
    """
    when fixing upper body
    """
    # lower_joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "left_hip_pitch_joint", 
    #         "left_hip_roll_joint", 
    #         "left_hip_yaw_joint", 
    #         "left_knee_joint", 
    #         "left_ankle_pitch_joint", 
    #         "left_ankle_roll_joint", 
            
    #         "right_hip_pitch_joint", 
    #         "right_hip_roll_joint", 
    #         "right_hip_yaw_joint", 
    #         "right_knee_joint", 
    #         "right_ankle_pitch_joint", 
    #         "right_ankle_roll_joint", 
    #     ],
    #     scale={
    #         "left_hip_pitch_joint": 0.25, 
    #         "left_hip_roll_joint": 0.25, 
    #         "left_hip_yaw_joint": 0.25, 
    #         "left_knee_joint": 0.25, 
    #         "left_ankle_pitch_joint": 0.25, 
    #         "left_ankle_roll_joint": 0.25, 
            
    #         "right_hip_pitch_joint": 0.25, 
    #         "right_hip_roll_joint": 0.25, 
    #         "right_hip_yaw_joint": 0.25, 
    #         "right_knee_joint": 0.25, 
    #         "right_ankle_pitch_joint": 0.25, 
    #         "right_ankle_roll_joint": 0.25, 
    #     },
    #     preserve_order=True,
    #     use_default_offset=True,
    # )
    
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        # backend="2D",
        backend="3D",
        intruder_geometry_cfg=G1FootGeometryCfg(),
        enable_ema_filter=True,
    )