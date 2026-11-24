# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for the `newton_box` task, replicating Newton's `example_mpm_twoway_coupling`.

Six rigid boxes (75 kg, friction 0.5) are dropped from 2.0 m, spaced 0.6 m apart vertically, onto a
2 m x 2 m x 0.5 m sand bed resting on a ground plane at z = 0. The sand uses the training-set material
(friction 0.48, density 1600 kg/m^3); the example uses 0.75 and 2500.
Box sizes are the example's half extents doubled. The example emits 3 particles per 0.05 m voxel
(about 430k particles); here the particle spacing is that of the Particle NeRD Local training data
(0.032 m), which gives about 63k.
"""

import numpy as np
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.g1_29dof_mpm.env_cfg.scene_cfg import MPM_VISUAL_COLOR

VOXEL_SIZE = 0.04
SPACING = VOXEL_SIZE / 1.25
SAND_DENSITY = 1600.0
"""Training-set sand density; the Newton example uses 2500 kg/m^3."""
SAND_FRICTION = 0.48
"""Training-set sand friction; the Newton example uses 0.75."""
BED_LO = (-1.0, -1.0, 0.0)
BED_HI = (1.0, 1.0, 0.5)
PARTICLE_JITTER = 0.20 * SPACING
PARTICLE_SEED = 42
DROP_Z = 2.0
Z_SEPARATION = 0.6
BOX_MASS = 75.0
BOX_HALF_EXTENTS = (
    (0.25, 0.35, 0.25),
    (0.25, 0.25, 0.25),
    (0.30, 0.20, 0.20),
    (0.25, 0.35, 0.25),
    (0.25, 0.25, 0.25),
    (0.30, 0.20, 0.20),
)
BOX_OFFSETS_XY = ((0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1), (0.1, 0.1))
BOX_COLORS = (
    (0.80, 0.30, 0.25),
    (0.25, 0.55, 0.80),
    (0.30, 0.70, 0.40),
    (0.85, 0.65, 0.20),
    (0.60, 0.40, 0.75),
    (0.30, 0.70, 0.70),
)


def bed_positions() -> list[list[float]]:
    """Jittered particle lattice filling the sand bed."""
    lo, hi = np.array(BED_LO), np.array(BED_HI)
    count = np.ceil((hi - lo) / SPACING).astype(int)
    axes = [lo[i] + (np.arange(count[i]) + 0.5) * (hi[i] - lo[i]) / count[i] for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(PARTICLE_SEED)
    points = points + rng.uniform(-PARTICLE_JITTER, PARTICLE_JITTER, size=points.shape)
    points = np.clip(points, lo + 0.5 * SPACING, hi)
    return points.astype(np.float32).tolist()


def _boundary_cfg(
    name: str, size: tuple[float, float, float], pos: tuple[float, float, float], visible: bool
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionCfg(contact_margin=0.016),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.55, 0.58, 0.60)),
            visible=visible,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _box_cfg(index: int) -> RigidObjectCfg:
    hx, hy, hz = BOX_HALF_EXTENTS[index]
    ox, oy = BOX_OFFSETS_XY[index]
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box" + str(index),
        spawn=sim.CuboidCfg(
            size=(2 * hx, 2 * hy, 2 * hz),
            rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
            mass_props=sim.MassCfg(mass=BOX_MASS),
            collision_props=NewtonCollisionCfg(contact_margin=0.0),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=BOX_COLORS[index]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(ox, oy, DROP_Z + index * Z_SEPARATION)),
    )


@configclass
class NewtonBoxSceneCfg(InteractiveSceneCfg):
    """Six falling boxes, a sand bed and a ground plane. Boundary walls sit 10 m away so the ground is open."""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    box0: RigidObjectCfg = _box_cfg(0)
    box1: RigidObjectCfg = _box_cfg(1)
    box2: RigidObjectCfg = _box_cfg(2)
    box3: RigidObjectCfg = _box_cfg(3)
    box4: RigidObjectCfg = _box_cfg(4)
    box5: RigidObjectCfg = _box_cfg(5)

    sand = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMPointsCfg(
            positions=bed_positions(),
            radius=0.5 * SPACING,
            mass=SAND_DENSITY * (SPACING**3),
            visual_color=MPM_VISUAL_COLOR,
            material=MPMParticleMaterialCfg(
                density=SAND_DENSITY,
                young_modulus=1.0e15,
                poisson_ratio=0.30,
                friction=SAND_FRICTION,
                yield_pressure=1.0e15,
                tensile_yield_ratio=0.0,
                yield_stress=0.0,
                hardening=0.0,
                dilatancy=0.0,
                viscosity=0.0,
            ),
        ),
    )

    def __post_init__(self) -> None:
        wall = 0.5
        self.floor = _boundary_cfg("Floor", (30.0, 30.0, 0.20), (0.0, 0.0, -0.10), True)
        self.wall_x0 = _boundary_cfg("WallX0", (0.10, 30.0, wall), (-10.0, 0.0, 0.0), False)
        self.wall_x1 = _boundary_cfg("WallX1", (0.10, 30.0, wall), (10.0, 0.0, 0.0), False)
        self.wall_y0 = _boundary_cfg("WallY0", (30.0, 0.10, wall), (0.0, -10.0, 0.0), False)
        self.wall_y1 = _boundary_cfg("WallY1", (30.0, 0.10, wall), (0.0, 10.0, 0.0), False)
        self.rigid_floor = _boundary_cfg("RigidFloor", (30.0, 30.0, 0.20), (0.0, 0.0, -0.10), False)
        self.rigid_wall_x0 = _boundary_cfg("RigidWallX0", (0.10, 30.0, wall), (-10.0, 0.0, 0.0), False)
        self.rigid_wall_x1 = _boundary_cfg("RigidWallX1", (0.10, 30.0, wall), (10.0, 0.0, 0.0), False)
        self.rigid_wall_y0 = _boundary_cfg("RigidWallY0", (30.0, 0.10, wall), (0.0, -10.0, 0.0), False)
        self.rigid_wall_y1 = _boundary_cfg("RigidWallY1", (30.0, 0.10, wall), (0.0, 10.0, 0.0), False)
