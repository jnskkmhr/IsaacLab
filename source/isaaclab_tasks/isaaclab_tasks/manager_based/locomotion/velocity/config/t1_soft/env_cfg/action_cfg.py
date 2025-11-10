# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_soft.mdp as t1_mdp

@configclass
class T1ActionsCfg:
    """Action specifications for the MDP."""

    lower_joint_pos = mdp.JointPositionActionCfg(  # type: ignore
        asset_name="robot",
        joint_names=[
                "Waist",
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
            "Waist": 0.0,
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
        offset={
            "Waist": 0.0,
            "Left_Hip_Pitch": -0.2,
            "Left_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.4,
            "Left_Ankle_Pitch": -0.25,
            "Left_Ankle_Roll": 0.0,
            "Right_Hip_Pitch": -0.2,
            "Right_Hip_Roll": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Right_Knee_Pitch": 0.4,
            "Right_Ankle_Pitch": -0.25,
            "Right_Ankle_Roll": 0.0,
        },
        preserve_order=True,
        use_default_offset=False,
    )
    
    # joint_pos = mdp.JointPositionActionCfg( # type: ignore
    #     asset_name="robot", 
    #     joint_names=[".*"], 
    #     scale=0.5, 
    #     use_default_offset=True, 
    #     )
    
    physics_callback = t1_mdp.PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_foot_link"],
        max_terrain_level=10, # maximum stiffness scaling
        backend="3D",
    )