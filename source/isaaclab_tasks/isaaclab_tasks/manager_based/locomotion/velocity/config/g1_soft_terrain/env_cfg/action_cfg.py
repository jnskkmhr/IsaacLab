# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_soft_terrain.mdp as g1_mdp

@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg( # type: ignore
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.5, 
        use_default_offset=True, 
        ) 
    
    physics_callback = g1_mdp.PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        max_terrain_level=5,
    )