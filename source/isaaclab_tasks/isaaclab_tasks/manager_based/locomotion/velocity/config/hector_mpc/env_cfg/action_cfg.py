# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector_mpc.mdp as hector_mdp
from isaaclab_tasks.manager_based.soft_contact import IntruderGeometryCfg, PhysicsCallbackActionCfg


@configclass
class HECTORFootGeometryCfg(IntruderGeometryCfg):
    """Configuration for the intruder geometry used in soft contact modeling."""
    contact_edge_x: tuple[float, float] = (-0.054, 0.091)  # length in x direction (m)
    contact_edge_y: tuple[float, float] = (-0.0365, 0.0365)  # length in y direction (m)
    contact_edge_z: tuple[float, float] = (-0.0486, 0.0)  # length in z direction (m)
    num_contact_points: int = 10 * 10

@configclass
class HECTORActionsCfg:
    """Action specifications for the MDP."""
    
    mpc_action = hector_mdp.BlindLocomotionMPCActionCfgResAll(
        asset_name="robot", 
        joint_names=['L_hip_joint','L_hip2_joint','L_thigh_joint','L_calf_joint','L_toe_joint', 'R_hip_joint','R_hip2_joint','R_thigh_joint','R_calf_joint','R_toe_joint'],
        action_range = (
            (-2.0, -2.0, -4.0,    -1.0, -1.0, -1.0,    -0.2/13.856, -0.2/13.856, -0.2/13.856,    -0.2/0.5413, -0.2/0.52, -0.2/0.0691,   -0.25, -0.15, -0.66), 
            (2.0, 2.0, 4.0,        1.0, 1.0, 1.0,       0.2/13.856,  0.2/13.856,  0.2/13.856,     0.2/0.5413,  0.2/0.52,  0.2/0.0691,    0.25, 0.15, 0.66)
        ), 
        negative_action_clip_idx=[13],
        # debug_vis=True,
    )
    
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_toe"],
        max_terrain_level=5,
        backend="3D",
        intruder_geometry_cfg=HECTORFootGeometryCfg(),
    )