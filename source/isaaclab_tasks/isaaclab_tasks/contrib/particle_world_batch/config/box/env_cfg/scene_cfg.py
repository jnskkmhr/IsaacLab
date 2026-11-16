# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone scene configuration for the `box` two-way coupling task (`example_mpm_twoway_coupling.py`)."""

from copy import deepcopy

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Physical parameters matching `newton/examples/mpm/example_mpm_twoway_coupling.py`
BOX_VOXEL_SIZE = 0.02
BOX_PARTICLES_PER_CELL = 1.25
BOX_BED_SIZE = (2.0, 1.2, 0.30)
BOX_DENSITY = 2500.0
BOX_FRICTION = 0.75
BOX_MASS = 75.0
BOX_HALF_EXTENTS = (0.12, 0.09, 0.06)

BOX_SAND_MATERIAL_CFG = MPMParticleMaterialCfg(
    density=BOX_DENSITY,
    young_modulus=1.0e15,
    poisson_ratio=0.3,
    friction=BOX_FRICTION,
    yield_pressure=1.0e15,
    tensile_yield_ratio=0.0,
    yield_stress=0.0,
    hardening=0.0,
    dilatancy=0.0,
    viscosity=0.0,
)


def _sand_bed_cfg(
    size: tuple[float, float, float] = BOX_BED_SIZE,
    voxel_size: float = BOX_VOXEL_SIZE,
    particles_per_cell: float = BOX_PARTICLES_PER_CELL,
) -> MPMObjectCfg:
    """Create a declarative `MPMGridCfg` particle bed with full MPM parameter flexibility."""
    length, width, depth = size
    spacing = voxel_size / particles_per_cell
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMGridCfg(
            lower=(-length / 2.0, -width / 2.0, -depth),
            upper=(length / 2.0, width / 2.0, 0.0),
            voxel_size=voxel_size,
            particles_per_cell=particles_per_cell,
            particle_placement="cell_center",
            jitter=0.40 * spacing,
            # visual_color=(0.70, 0.60, 0.40),
            visual_color=(0.32, 0.24, 0.21),
            material=BOX_SAND_MATERIAL_CFG,
        ),
    )


def _tray_wall_cfg(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionCfg(contact_margin=0.016),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.25, 0.30, 0.33)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


@configclass
class BoxSceneCfg(InteractiveSceneCfg):
    """Scene spawning a dynamic rigid box dropped onto a granular sand bed."""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim.DomeLightCfg(
            intensity=1500.0,
            color=(0.85, 0.88, 0.95),
        ),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/distantLight",
        spawn=sim.DistantLightCfg(
            intensity=2500.0,
            color=(1.0, 0.98, 0.92),
            angle=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3826834, 0.0, 0.0, 0.9238795)),
    )

    sand: MPMObjectCfg = _sand_bed_cfg(BOX_BED_SIZE, BOX_VOXEL_SIZE, BOX_PARTICLES_PER_CELL)

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim.CuboidCfg(
            size=(2.0 * BOX_HALF_EXTENTS[0], 2.0 * BOX_HALF_EXTENTS[1], 2.0 * BOX_HALF_EXTENTS[2]),
            rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
            mass_props=sim.MassCfg(mass=BOX_MASS),
            collision_props=NewtonCollisionCfg(contact_margin=0.0),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.12, 0.32, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
    )

    def __post_init__(self):
        lower, upper = self.sand.spawn.lower, self.sand.spawn.upper
        length = float(upper[0] - lower[0])
        width = float(upper[1] - lower[1])
        depth = float(upper[2] - lower[2])
        wall_top = 0.20
        wall_height = depth + wall_top
        wall_center = (wall_top - depth) / 2.0

        self.floor = _tray_wall_cfg("Floor", (length + 0.2, width + 0.2, 0.20), (0.0, 0.0, -depth - 0.10))
        self.wall_x0 = _tray_wall_cfg(
            "WallX0", (0.10, width + 0.2, wall_height), (-length / 2 - 0.05, 0.0, wall_center)
        )
        self.wall_x1 = _tray_wall_cfg("WallX1", (0.10, width + 0.2, wall_height), (length / 2 + 0.05, 0.0, wall_center))
        self.wall_y0 = _tray_wall_cfg("WallY0", (length, 0.10, wall_height), (0.0, -width / 2 - 0.05, wall_center))
        self.wall_y1 = _tray_wall_cfg("WallY1", (length, 0.10, wall_height), (0.0, width / 2 + 0.05, wall_center))

        for name in ("floor", "wall_x0", "wall_x1", "wall_y0", "wall_y1"):
            mirror = deepcopy(getattr(self, name))
            mirror.prim_path = mirror.prim_path.replace("}/", "}/Rigid")
            mirror.spawn.visible = False
            setattr(self, "rigid_" + name, mirror)
