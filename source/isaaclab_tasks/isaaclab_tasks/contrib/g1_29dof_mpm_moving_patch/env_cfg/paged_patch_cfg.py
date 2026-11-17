# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Particle-slot paging parameters for the generated-terrain task."""

from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

from isaaclab.utils import configclass

from .particle_cache_cfg import ParticleCacheCfg


@configclass
class PagedPatchCfg:
    simulated_terrain_size: tuple[float, float] = (1.2, 1.2)
    boundary_terrain_size: float = 0.2
    tracked_body: str = "pelvis"
    tracked_body_patterns: tuple[str, ...] = ()
    patch_discretization_step: float = 0.2
    particle_depth: float = 0.3
    voxel_size: float = 0.02
    particles_per_cell: float = 1.25
    jitter: float = 0.016
    material: MPMParticleMaterialCfg | None = None
    visual_color: tuple[float, float, float] = (0.32, 0.24, 0.21)
    particle_cache: ParticleCacheCfg = ParticleCacheCfg(max_cached_particles_per_env=2_000_000)

    @property
    def total_patch_size(self) -> tuple[float, float]:
        return tuple(size + 2 * self.boundary_terrain_size for size in self.simulated_terrain_size)
