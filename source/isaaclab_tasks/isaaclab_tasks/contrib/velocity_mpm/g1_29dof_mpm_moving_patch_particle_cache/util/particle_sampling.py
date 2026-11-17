# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Use IsaacLab's grid sampling for both initial and newly visited terrain tiles."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMPointsCfg
from isaaclab_newton.sim.spawners.mpm.mpm import _generate_grid_particles, spawn_mpm_particles

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .terrain import MovingPatchTerrainCfg


def _sample_grid_tile(cfg: CachedMPMGridCfg) -> tuple[np.ndarray, float, float]:
    size = cfg.tile_size
    positions, _, masses, radii = _generate_grid_particles(
        cfg.replace(lower=(0.0, 0.0, cfg.lower[2]), upper=(size, size, cfg.upper[2]))
    )
    # Keep jittered samples in their birth tile without clipping their distribution.
    positions[:, :2] %= size
    return positions, float(masses[0]), float(radii[0])


def spawn_cached_particles(prim_path, cfg: CachedMPMGridCfg, translation=None, orientation=None, **kwargs):
    """Generate reserve slots at scene creation, keeping large particle arrays out of serialized configs."""
    template, mass, radius = _sample_grid_tile(cfg)
    # Duplicate positions only reserve storage. Unused slots become inactive before physics runs.
    positions = np.tile(template, (cfg.get_particle_count() // len(template), 1))
    points_cfg = MPMPointsCfg(
        visible=cfg.visible,
        positions=positions,
        mass=mass,
        radius=radius,
        material=cfg.material,
        visual_color=cfg.visual_color,
        visual_material=cfg.visual_material,
        visual_update_frequency=cfg.visual_update_frequency,
    )
    return spawn_mpm_particles(prim_path, points_cfg, translation=translation, orientation=orientation, **kwargs)


@configclass
class CachedMPMGridCfg(MPMGridCfg):
    """Standard grid parameters with reserve slots for variable particle accumulation."""

    func = spawn_cached_particles
    tile_size: float = MISSING
    """Terrain tile side length [m], supplied by the scene's particle-cache configuration."""
    simulation_capacity_multiplier: int = MISSING
    """Simulation slots divided by initial particle count, supplied by the cache configuration."""

    def get_particle_count(self) -> int:
        """Number of fixed simulation slots, including inactive reserves."""
        template, _, _ = _sample_grid_tile(self)
        dimensions = np.rint((np.asarray(self.upper[:2]) - self.lower[:2]) / self.tile_size).astype(int)
        return len(template) * int(np.prod(dimensions)) * self.simulation_capacity_multiplier


def make_particle_spawner(terrain: MovingPatchTerrainCfg) -> CachedMPMGridCfg:
    """Derive the emitter from the scene's final terrain and cache configuration."""
    x, y = terrain.total_patch_size
    return CachedMPMGridCfg(
        lower=(-x / 2, -y / 2, -terrain.particle_depth),
        upper=(x / 2, y / 2, 0.0),
        voxel_size=terrain.voxel_size,
        particles_per_cell=terrain.particles_per_cell,
        particle_placement="cell_center",
        jitter=terrain.jitter,
        material=terrain.material,
        visual_color=terrain.visual_color,
        tile_size=terrain.particle_cache.tile_size,
        simulation_capacity_multiplier=terrain.particle_cache.simulation_capacity_multiplier,
    )


def sample_particle_tile(terrain: MovingPatchTerrainCfg) -> tuple[np.ndarray, float, float]:
    """Return local particle positions [m], mass [kg] and radius [m] for one fresh tile."""
    return _sample_grid_tile(make_particle_spawner(terrain))
