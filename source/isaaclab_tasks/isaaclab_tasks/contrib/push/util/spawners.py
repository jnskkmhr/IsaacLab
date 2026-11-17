# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot collision geometry for whole-body pushing."""

from isaaclab_newton.sim.schemas import NewtonCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsMeshCollisionCfg


@sim_utils.clone
def spawn_robot_with_hand_colliders(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Enable the asset's rubber-hand meshes locally before cloning the robot.

    File-spawner collision overrides only modify existing colliders. These hand
    meshes are visual-only in the locomotion USD, so their APIs must be created.
    """
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    hand_paths = [
        str(hand.GetPath())
        for hand in sim_utils.find_matching_prims(f"{prim_path}/.*_rubber_hand")
        if hand.IsInstance()
    ]
    for path in hand_paths:
        sim_utils.make_uninstanceable(path)
    hand_meshes = f"{prim_path}/.*_rubber_hand/(left|right)_rubber_hand"
    if not sim_utils.apply_collision_properties(
        hand_meshes,
        [UsdPhysicsCollisionCfg(), NewtonCollisionCfg(contact_margin=0.003, contact_gap=0.0)],
        create_if_missing=True,
    ):
        raise RuntimeError("Could not create G1 rubber-hand colliders.")
    for mesh in sim_utils.find_matching_prims(hand_meshes):
        sim_utils.apply_mesh_collision_properties(
            str(mesh.GetPath()), [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")]
        )
    return prim
