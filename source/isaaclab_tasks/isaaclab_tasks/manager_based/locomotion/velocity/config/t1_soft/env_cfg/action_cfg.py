# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from isaaclab_tasks.manager_based.soft_contact import IntruderGeometryCfg, PhysicsCallbackActionCfg


@configclass
class T1FootGeometryCfg(IntruderGeometryCfg):
    """Configuration for the intruder geometry used in soft contact modeling."""
    contact_edge_x: tuple[float, float] = (-0.1021, 0.1228)  # length in x direction (m)
    contact_edge_y: tuple[float, float] = (-0.04793, 0.04793)  # length in y direction (m)
    contact_edge_z: tuple[float, float] = (-0.0305, 0.0)  # length in z direction (m)
    num_contact_points: int = 5 * 5
    # num_contact_points: int = 20 * 20 # inference

@configclass
class T1ActionsCfg:
    """Action specifications for the MDP."""

    lower_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
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
        ],
        scale={
            "Left_Hip_Pitch": 1.0,
            "Left_Hip_Roll": 1.0,
            "Left_Hip_Yaw": 1.0,
            "Left_Knee_Pitch": 1.0,
            "Left_Ankle_Pitch": 1.0,   # constrained ankle pitch
            "Left_Ankle_Roll": 1.0,   # constrained ankle roll

            "Right_Hip_Pitch": 1.0,
            "Right_Hip_Roll": 1.0,
            "Right_Hip_Yaw": 1.0,
            "Right_Knee_Pitch": 1.0,
            "Right_Ankle_Pitch": 1.0,   # constrained ankle pitch
            "Right_Ankle_Roll": 1.0,   # constrained ankle roll
        },
        preserve_order=True,
        use_default_offset=True,
    )
    
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_foot_link"],
        max_terrain_level=10, # maximum stiffness scaling
        backend="3D",
        intruder_geometry_cfg=T1FootGeometryCfg(),
        # enable_ema_filter=False,
    )