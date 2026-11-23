# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene configuration for the `particle` task: a block of sand dropped or tossed onto open flat ground."""

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
PARTICLE_JITTER = 0.20 * SPACING
PARTICLE_SEED = 42
BLOCK_SIZE = 0.40
RELEASE_HEIGHT = 0.20
FLOOR_TOP_Z = -0.30
BLOCK_CENTER_X = -0.15
"""World-frame x of the block centre; matches the free-fall training data."""


def block_positions(shape: str) -> list[list[float]]:
    """Jittered particle lattice for a `cube`, `sphere` or `cylinder` block released above the floor."""
    radius = SPACING / 2.0
    count = max(6, round(BLOCK_SIZE / SPACING))
    xs = (np.arange(count) - (count - 1) / 2) * SPACING + BLOCK_CENTER_X
    ys = (np.arange(count) - (count - 1) / 2) * SPACING
    z_bottom = FLOOR_TOP_Z + radius + RELEASE_HEIGHT
    zs = np.arange(count) * SPACING + z_bottom
    points = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    half = 0.5 * BLOCK_SIZE
    if shape == "cylinder":
        points = points[(points[:, 0] - BLOCK_CENTER_X) ** 2 + points[:, 1] ** 2 <= half**2]
    elif shape == "sphere":
        z_center = z_bottom + 0.5 * (count - 1) * SPACING
        distance_sq = (points[:, 0] - BLOCK_CENTER_X) ** 2 + points[:, 1] ** 2 + (points[:, 2] - z_center) ** 2
        points = points[distance_sq <= half**2]
    elif shape != "cube":
        raise ValueError(f"Unknown block shape: {shape!r}")
    rng = np.random.default_rng(PARTICLE_SEED)
    points = points + rng.uniform(-PARTICLE_JITTER, PARTICLE_JITTER, size=points.shape)
    points[:, 2] = np.maximum(points[:, 2], z_bottom)
    return points.astype(np.float32).tolist()


def _boundary_cfg(name: str, size: tuple[float, float, float], pos: tuple[float, float, float], visible: bool):
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


@configclass
class ParticleSceneCfg(InteractiveSceneCfg):
    """Open flat ground with a free sand block.

    The solver expects one floor and four walls. The walls sit 10 m away so the ground is effectively open,
    as in the free-fall training data. A hidden kinematic `anchor` body 10 m above the ground fills the rigid
    collider slot the model was trained with and never touches the sand.
    """

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Anchor",
        spawn=sim.CuboidCfg(
            size=(0.24, 0.18, 0.12),
            rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim.MassCfg(mass=2.0),
            collision_props=NewtonCollisionCfg(contact_margin=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
    )

    def _add_sand(self, shape: str) -> None:
        self.sand = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMPointsCfg(
                positions=block_positions(shape),
                radius=0.5 * SPACING,
                mass=SAND_DENSITY * (SPACING**3),
                visual_color=MPM_VISUAL_COLOR,
                material=MPMParticleMaterialCfg(
                    density=SAND_DENSITY,
                    young_modulus=1.0e15,
                    poisson_ratio=0.30,
                    friction=0.48,
                    yield_pressure=1.0e15,
                    tensile_yield_ratio=0.0,
                    yield_stress=0.0,
                    hardening=0.0,
                    dilatancy=0.0,
                    viscosity=0.0,
                ),
            ),
        )

    def __post_init__(self):
        wall_height = 0.5
        self.floor = _boundary_cfg("Floor", (4.0, 4.0, 0.20), (0.0, 0.0, FLOOR_TOP_Z - 0.10), True)
        self.wall_x0 = _boundary_cfg("WallX0", (0.10, 4.0, wall_height), (-10.0, 0.0, 0.0), False)
        self.wall_x1 = _boundary_cfg("WallX1", (0.10, 4.0, wall_height), (10.0, 0.0, 0.0), False)
        self.wall_y0 = _boundary_cfg("WallY0", (4.0, 0.10, wall_height), (0.0, -10.0, 0.0), False)
        self.wall_y1 = _boundary_cfg("WallY1", (4.0, 0.10, wall_height), (0.0, 10.0, 0.0), False)
        # Rigid-solver copies of the same boundaries, never drawn.
        self.rigid_floor = _boundary_cfg("RigidFloor", (4.0, 4.0, 0.20), (0.0, 0.0, FLOOR_TOP_Z - 0.10), False)
        self.rigid_wall_x0 = _boundary_cfg("RigidWallX0", (0.10, 4.0, wall_height), (-10.0, 0.0, 0.0), False)
        self.rigid_wall_x1 = _boundary_cfg("RigidWallX1", (0.10, 4.0, wall_height), (10.0, 0.0, 0.0), False)
        self.rigid_wall_y0 = _boundary_cfg("RigidWallY0", (4.0, 0.10, wall_height), (0.0, -10.0, 0.0), False)
        self.rigid_wall_y1 = _boundary_cfg("RigidWallY1", (4.0, 0.10, wall_height), (0.0, 10.0, 0.0), False)


@configclass
class ParticleCubeSceneCfg(ParticleSceneCfg):
    """Cube-shaped sand block."""

    def __post_init__(self):
        super().__post_init__()
        self._add_sand("cube")


@configclass
class ParticleSphereSceneCfg(ParticleSceneCfg):
    """Sphere-shaped sand block."""

    def __post_init__(self):
        super().__post_init__()
        self._add_sand("sphere")


@configclass
class ParticleCylinderSceneCfg(ParticleSceneCfg):
    """Cylinder-shaped sand block with a vertical axis."""

    def __post_init__(self):
        super().__post_init__()
        self._add_sand("cylinder")
