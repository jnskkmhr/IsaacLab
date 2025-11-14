# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_soft_terrain.mdp as g1_mdp
from isaaclab_tasks.manager_based.soft_contact import IntruderGeometryCfg, PhysicsCallbackActionCfg

@configclass    
class G1FootGeometryCfg(IntruderGeometryCfg):
    """Configuration for the intruder geometry used in soft contact modeling."""
    contact_edge_x: tuple[float, float] = (-0.065, 0.141)  # length in x direction (m)
    contact_edge_y: tuple[float, float] = (-0.0368, 0.0368)  # length in y direction (m)
    contact_edge_z: tuple[float, float] = (-0.039, 0.0)  # length in z direction (m)
    num_contact_points: int = 5 * 5
    # num_contact_points: int = 20 * 20 # inference

@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg( # type: ignore
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.5, 
        use_default_offset=True, 
        ) 
    
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        max_terrain_level=10,
        backend="3D",
        intruder_geometry_cfg=G1FootGeometryCfg(),
        enable_ema_filter=True,
    )