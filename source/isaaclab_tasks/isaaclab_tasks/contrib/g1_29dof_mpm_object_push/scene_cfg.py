# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat sand supporting both G1 and a dynamic box."""

from isaaclab_newton.sim.schemas import NewtonCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsMeshCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.scene_cfg import G1MovingPatchSceneCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.terrain_cfg import FLAT_TERRAINS_CFG

BOX_SIZE = (0.5, 0.65, 0.7)


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


@configclass
class G1ObjectPushSceneCfg(G1MovingPatchSceneCfg):
    """A 5 kg box inside the robot-centered dynamic particle region."""

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.0, 0.36)),
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=UsdPhysicsRigidBodyCfg(),
            collision_props=[UsdPhysicsCollisionCfg(), NewtonCollisionCfg(contact_margin=0.005, contact_gap=0.0)],
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.75)),
        ),
    )

    def __post_init__(self):
        self.robot.spawn.func = spawn_robot_with_hand_colliders
        self.terrain.terrain_generator = FLAT_TERRAINS_CFG.copy()
        self.terrain.use_terrain_origins = False
        self.terrain.moving_patch_terrain.simulated_terrain_size = (3.6, 2.4)
        self.terrain.moving_patch_terrain.jitter = 0.004
        super().__post_init__()
