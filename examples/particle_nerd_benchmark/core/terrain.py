# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Finite sand beds with a rigid floor and retaining walls, in metres."""

import numpy as np
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg

VOXEL_SIZE = 0.04
SPACING = VOXEL_SIZE / 1.25
DENSITY = 1600.0
PARTICLE_JITTER = 0.20 * SPACING
PARTICLE_SEED = 42


def surface_height(x, terrain):
    """Initial particle free-surface height relative to the tray centre."""
    if terrain == "flat":
        return np.zeros_like(x)
    if terrain == "ascend":
        return 0.12 * np.asarray(x)
    if terrain == "descend":
        return -0.12 * np.asarray(x)
    if terrain == "wave":
        return 0.06 * np.sin(2 * np.pi * np.asarray(x) / 1.2)
    raise ValueError(f"Unknown terrain: {terrain}")


def sand_cfg(terrain="flat", size=(2.0, 1.2, 0.30), *, seed=PARTICLE_SEED):
    """Jitter lattice samples within the initial bed, preserving count and mass.

    Each coordinate is sampled uniformly within 20% of the particle spacing
    around its lattice position, truncated at the rigid tray and initial sand
    surface. The fixed seed gives reproducible initial positions across runs.
    """
    length, width, depth = size
    axes = [
        np.arange(-length / 2 + SPACING / 2, length / 2, SPACING),
        np.arange(-width / 2 + SPACING / 2, width / 2, SPACING),
        np.arange(-depth + SPACING / 2, 0.2, SPACING),
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = points[points[:, 2] < surface_height(points[:, 0], terrain)]
    rng = np.random.default_rng(seed)
    lower = np.maximum(points - PARTICLE_JITTER, (-length / 2, -width / 2, -depth))
    upper = np.minimum(points + PARTICLE_JITTER, (length / 2, width / 2, np.inf))
    points[:, :2] = rng.uniform(lower[:, :2], upper[:, :2])
    upper[:, 2] = np.minimum(upper[:, 2], surface_height(points[:, 0], terrain))
    points[:, 2] = rng.uniform(lower[:, 2], upper[:, 2])
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMPointsCfg(
            positions=points.tolist(),
            radius=SPACING / 2,
            mass=DENSITY * SPACING**3,
            visual_color=(0.72, 0.59, 0.36),
            material=MPMParticleMaterialCfg(
                density=DENSITY,
                young_modulus=1e15,
                poisson_ratio=0.3,
                friction=0.48,
                yield_pressure=1e15,
                tensile_yield_ratio=0.0,
                yield_stress=0.0,
                hardening=0.0,
                dilatancy=0.0,
                viscosity=0.0,
            ),
        ),
    )


def tray_box(name, size, position):
    """Static collision geometry visible in both rigid and particle solvers."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionCfg(contact_margin=0.016),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.25, 0.30, 0.33)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


def configure_terrain(scene, terrain, size=(2.0, 1.2, 0.30), *, wall_top=0.20):
    """Assign the particle bed and five static retaining boxes."""
    length, width, depth = size
    scene.sand = sand_cfg(terrain, size)
    scene.floor = tray_box("Floor", (length + 0.2, width + 0.2, 0.20), (0.0, 0.0, -depth - 0.10))
    wall_height = depth + wall_top
    wall_centre = (wall_top - depth) / 2
    scene.wall_x0 = tray_box("WallX0", (0.10, width + 0.2, wall_height), (-length / 2 - 0.05, 0.0, wall_centre))
    scene.wall_x1 = tray_box("WallX1", (0.10, width + 0.2, wall_height), (length / 2 + 0.05, 0.0, wall_centre))
    scene.wall_y0 = tray_box("WallY0", (length, 0.10, wall_height), (0.0, -width / 2 - 0.05, wall_centre))
    scene.wall_y1 = tray_box("WallY1", (length, 0.10, wall_height), (0.0, width / 2 + 0.05, wall_centre))
    # Separate static-shape ownership: each solver receives its own coincident collider.
    from copy import deepcopy

    for name in ("floor", "wall_x0", "wall_x1", "wall_y0", "wall_y1"):
        mirror = deepcopy(getattr(scene, name))
        mirror.prim_path = mirror.prim_path.replace("}/", "}/Rigid")
        mirror.spawn.visible = False
        setattr(scene, "rigid_" + name, mirror)
