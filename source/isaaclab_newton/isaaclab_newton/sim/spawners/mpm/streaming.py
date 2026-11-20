# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Particle spawner for a streaming MPM pool with a tile cache.

The same grid sampling produces the initial pool and the template of every tile generated later.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np

from isaaclab.utils import configclass

from .mpm import _generate_grid_particles, spawn_mpm_particles
from .mpm_cfg import MPMGridCfg, MPMPointsCfg


def sample_particle_tile(cfg: StreamingMPMGridCfg) -> tuple[np.ndarray, float, float]:
    """Return local particle positions [m], mass [kg] and radius [m] of one tile of the pool."""
    size = cfg.tile_size
    positions, _, masses, radii = _generate_grid_particles(
        cfg.replace(lower=(0.0, 0.0, cfg.lower[2]), upper=(size, size, cfg.upper[2]))
    )
    # Keep jittered samples in their birth tile without clipping their distribution.
    positions[:, :2] %= size
    return positions, float(masses[0]), float(radii[0])


def spawn_streaming_mpm_particles(prim_path, cfg: StreamingMPMGridCfg, translation=None, orientation=None, **kwargs):
    """Create the pool's simulation slots, including the spare ones a tile cache needs.

    Large particle arrays are generated here, at scene creation, to keep them out of serialized configs.
    """
    template, mass, radius = sample_particle_tile(cfg)
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
class StreamingMPMGridCfg(MPMGridCfg):
    """Grid of a streaming pool's footprint, with spare slots for sand that accumulates in a tile."""

    func = spawn_streaming_mpm_particles

    tile_size: float = MISSING
    """Tile side length [m], taken from the solver's ``StreamingCfg``."""

    simulation_capacity_multiplier: float = MISSING
    """Simulation slots divided by the footprint's particle count, taken from ``ParticleCacheCfg``."""

    def get_particle_count(self) -> int:
        """Number of fixed simulation slots, including inactive spares."""
        template, _, _ = sample_particle_tile(self)
        dimensions = np.rint((np.asarray(self.upper[:2]) - self.lower[:2]) / self.tile_size).astype(int)
        # Spare storage is allocated in whole template tiles, so a fractional multiplier rounds up to a tile.
        tiles = int(np.ceil(int(np.prod(dimensions)) * self.simulation_capacity_multiplier))
        return len(template) * tiles
