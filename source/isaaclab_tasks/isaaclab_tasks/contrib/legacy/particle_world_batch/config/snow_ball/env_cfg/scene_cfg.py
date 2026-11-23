# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone scene configuration for the `snow_ball` slope rolling task (`example_mpm_snow_ball.py`)."""

from copy import deepcopy

import numpy as np
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, MPMPointsCfg
from isaaclab_newton.sim.spawners.mpm.mpm import _generate_grid_particles, spawn_mpm_particles

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

# Physical parameters matching `newton/examples/mpm/example_mpm_snow_ball.py`
SNOW_VOXEL_SIZE = 0.04
SNOW_PARTICLES_PER_CELL = 1.25
SNOW_BED_SIZE = (2.0, 1.2, 0.30)
SNOW_DENSITY = 400.0
SNOW_YOUNG_MODULUS = 1.4e6
SNOW_POISSON_RATIO = 0.3
SNOW_FRICTION = 0.5
SNOW_DAMPING = 0.01
SNOW_YIELD_PRESSURE = 1.4e6
SNOW_TENSILE_YIELD_RATIO = 0.2
SNOW_YIELD_STRESS = 0.0
SNOW_HARDENING = 5.0
SNOW_DILATANCY = 1.0
SNOW_BALL_RADIUS = 0.10
SNOW_BALL_MASS = 4.0

SNOW_MATERIAL_CFG = MPMParticleMaterialCfg(
    density=SNOW_DENSITY,
    young_modulus=SNOW_YOUNG_MODULUS,
    poisson_ratio=SNOW_POISSON_RATIO,
    friction=SNOW_FRICTION,
    damping=SNOW_DAMPING,
    yield_pressure=SNOW_YIELD_PRESSURE,
    tensile_yield_ratio=SNOW_TENSILE_YIELD_RATIO,
    yield_stress=SNOW_YIELD_STRESS,
    hardening=SNOW_HARDENING,
    dilatancy=SNOW_DILATANCY,
    viscosity=0.0,
)


def _snow_surface_height(x: np.ndarray, slope: float = -0.12, wave_amplitude: float = 0.015) -> np.ndarray:
    """Descending snow slope profile z(x) = slope * x + wave_amplitude * sin(2*pi*x / 1.0)."""
    x_arr = np.asarray(x)
    return slope * x_arr + wave_amplitude * np.sin(2.0 * np.pi * x_arr / 1.0)


@clone
def spawn_sloped_snow_mpm(
    prim_path: str,
    cfg: "SlopedSnowMPMGridCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
):
    """Generate a sloped snow lattice from `SlopedSnowMPMGridCfg`."""
    positions, velocities, masses, radii = _generate_grid_particles(cfg)
    depth = float(cfg.upper[2] - cfg.lower[2])
    min_z = float(cfg.lower[2]) + float(radii[0]) * 0.5
    positions[:, 2] = np.clip(
        positions[:, 2] + _snow_surface_height(positions[:, 0], cfg.slope, cfg.wave_amplitude),
        min_z,
        float(cfg.upper[2]) + 0.5 * depth,
    )
    points_cfg = MPMPointsCfg(
        positions=positions,
        velocities=velocities,
        mass=masses,
        radius=radii,
        material=cfg.material,
        visual_color=cfg.visual_color,
        visual_material=cfg.visual_material,
        visual_update_frequency=cfg.visual_update_frequency,
        visible=cfg.visible,
    )
    prim = spawn_mpm_particles.__wrapped__(
        prim_path, points_cfg, translation=translation, orientation=orientation, **kwargs
    )
    cfg.spawn_path = points_cfg.spawn_path
    return prim


@configclass
class SlopedSnowMPMGridCfg(MPMGridCfg):
    """Declarative `MPMGridCfg` with sinusoidal slope offset (`voxel_size`, `particles_per_cell`, `radius`, `mass`)."""

    func = spawn_sloped_snow_mpm
    slope: float = -0.12
    wave_amplitude: float = 0.015


def _snow_pack_cfg(
    size: tuple[float, float, float] = SNOW_BED_SIZE,
    voxel_size: float = SNOW_VOXEL_SIZE,
    particles_per_cell: float = SNOW_PARTICLES_PER_CELL,
) -> MPMObjectCfg:
    """Create a declarative `SlopedSnowMPMGridCfg` snow pack with full MPM parameter flexibility."""
    length, width, depth = size
    spacing = voxel_size / particles_per_cell
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=SlopedSnowMPMGridCfg(
            lower=(-length / 2.0, -width / 2.0, -depth),
            upper=(length / 2.0, width / 2.0, 0.0),
            voxel_size=voxel_size,
            particles_per_cell=particles_per_cell,
            particle_placement="cell_center",
            jitter=0.40 * spacing,
            # visual_color=(0.82, 0.85, 0.94),
            visual_color=(0.32, 0.24, 0.21),
            material=SNOW_MATERIAL_CFG,
        ),
    )


def _tray_wall_cfg(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionCfg(contact_margin=0.016),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.28, 0.33, 0.38)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


@configclass
class SnowBallSceneCfg(InteractiveSceneCfg):
    """Scene spawning a rolling snow-ball collider at the top of a cohesive snow slope (`example_mpm_snow_ball.py`)."""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim.DomeLightCfg(
            intensity=1500.0,
            color=(0.85, 0.90, 0.98),
        ),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/distantLight",
        spawn=sim.DistantLightCfg(
            intensity=2500.0,
            color=(1.0, 0.98, 0.95),
            angle=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3826834, 0.0, 0.0, 0.9238795)),
    )

    sand: MPMObjectCfg = _snow_pack_cfg(SNOW_BED_SIZE, SNOW_VOXEL_SIZE, SNOW_PARTICLES_PER_CELL)

    snow_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim.SphereCfg(
            radius=SNOW_BALL_RADIUS,
            rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
            mass_props=sim.MassCfg(mass=SNOW_BALL_MASS),
            collision_props=NewtonCollisionCfg(contact_margin=0.0),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.92, 0.95, 0.99)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.65, 0.0, 0.22),
            lin_vel=(0.35, 0.0, 0.0),
            ang_vel=(0.0, 3.5, 0.0),
        ),
    )

    def __post_init__(self):
        lower, upper = self.sand.spawn.lower, self.sand.spawn.upper
        length = float(upper[0] - lower[0])
        width = float(upper[1] - lower[1])
        depth = float(upper[2] - lower[2])
        wall_top = 0.25
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
