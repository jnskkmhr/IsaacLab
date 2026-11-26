# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The mixed tow task with a towed object that differs between environments: a box, a sphere or a wheel.

Every environment holds one of the three objects for the whole run; the scene's clone plan spreads them evenly
over the sand environments and over the rigid-ground environments. All three have the same mass, the same
height and the same reset pose, and follow the same mass curriculum. The box is pulled at the bottom of its
front face and slides; the sphere and the wheel (a cylinder lying on its side, axle across the tow direction)
are pulled at their centre and roll.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg
from isaaclab.utils import configclass

from .env_cfg.scene_cfg import SLED_SIZE
from .tow_mixed_env_cfg import G1TowMixedEnvCfg, G1TowMixedEnvCfg_PLAY

OBJECT_RADIUS = 0.5 * SLED_SIZE[2]
"""Radius of the sphere and of the wheel [m]: both stand as high as the box."""
WHEEL_WIDTH = SLED_SIZE[1]
"""Width of the wheel along its axle [m]."""
TETHER_OFFSETS = {
    "box": (0.5 * SLED_SIZE[0], 0.0, -0.5 * SLED_SIZE[2] + 0.05),
    "sphere": (0.0, 0.0, 0.0),
    "wheel": (0.0, 0.0, 0.0),
}
"""Attachment point of the tether in each object's frame [m]."""


OBJECT_NAMES = tuple(TETHER_OFFSETS)
"""Towed objects in the order of the spawner's variants."""


def object_spawners(box: sim_utils.CuboidCfg) -> list[sim_utils.SpawnerCfg]:
    """The three towed objects in the order of :data:`OBJECT_NAMES`, with the box's mass, collision and material.

    All three are meshes that collide by their convex hull: the rigid-body solver needs the same shape type in
    every environment and makes an exception only for meshes. Without the explicit approximation the box and the
    sphere would be imported as primitives.
    """
    shared = dict(
        mesh_collision_props=UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
        rigid_props=box.rigid_props,
        mass_props=box.mass_props,
        collision_props=box.collision_props,
        physics_material=box.physics_material,
    )
    return [
        sim_utils.MeshCuboidCfg(size=box.size, visual_material=box.visual_material, **shared),
        sim_utils.MeshSphereCfg(
            radius=OBJECT_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.25, 0.2), roughness=0.6),
            **shared,
        ),
        sim_utils.MeshCylinderCfg(
            radius=OBJECT_RADIUS,
            height=WHEEL_WIDTH,
            axis="Y",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.3), roughness=0.6),
            **shared,
        ),
    ]


@configclass
class G1TowObjectsEnvCfg(G1TowMixedEnvCfg):
    """The mixed tow task with a box, a sphere or a wheel behind each robot."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.sled.spawn = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=object_spawners(self.scene.sled.spawn), random_choice=False
        )


@configclass
class G1TowObjectsEnvCfg_PLAY(G1TowObjectsEnvCfg, G1TowMixedEnvCfg_PLAY):
    """Playback: one environment per object on sand and one per object on rigid ground."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 6
