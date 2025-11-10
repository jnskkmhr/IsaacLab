# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_legacy.mdp as t1_mdp

@configclass
class T1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg( # type: ignore
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.5, 
        use_default_offset=True, 
        )
    
    physics_callback = t1_mdp.PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_foot_link"],
        max_terrain_level=10,
        backend="3D",
    )