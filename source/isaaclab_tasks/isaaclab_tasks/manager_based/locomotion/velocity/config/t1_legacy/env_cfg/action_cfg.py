# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.soft_contact import IntruderGeometryCfg, PhysicsCallbackActionCfg

@configclass
class T1FootGeometryCfg(IntruderGeometryCfg):
    """Configuration for the intruder geometry used in soft contact modeling."""
    contact_edge_x: tuple[float, float] = (-0.1021, 0.1228)  # length in x direction (m)
    contact_edge_y: tuple[float, float] = (-0.04793, 0.04793)  # length in y direction (m)
    contact_edge_z: tuple[float, float] = (-0.0305, 0.0)  # length in z direction (m)
    num_contact_points: int = 10 * 10

@configclass
class T1ActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg( # type: ignore
    #     asset_name="robot", 
    #     joint_names=[".*"], 
    #     scale=0.5, 
    #     use_default_offset=True, 
    #     )
    
    joint_pos = mdp.JointPositionActionCfg(  # type: ignore
        asset_name="robot",
        joint_names=[
                # "Waist",
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",

                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",

                "Left_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Left_Elbow_Yaw",
                "Left_Wrist_Pitch",
                "Left_Wrist_Yaw",
                "Left_Hand_Roll",

                "Right_Shoulder_Pitch",
                "Right_Shoulder_Roll",
                "Right_Elbow_Pitch",
                "Right_Elbow_Yaw",
                "Right_Wrist_Pitch",
                "Right_Wrist_Yaw",
                "Right_Hand_Roll",
        ],
        scale=1.0,
        preserve_order=True,
        use_default_offset=True,
    )
    
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_foot_link"],
        max_terrain_level=10,
        backend="3D",
        intruder_geometry_cfg=T1FootGeometryCfg(),
    )