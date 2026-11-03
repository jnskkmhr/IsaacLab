# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Storage limits for independent terrain histories."""

from isaaclab.utils import configclass


@configclass
class ParticleCacheCfg:
    """Save departed particles on the GPU; clear an environment's history on reset."""

    tile_size: float = 0.2
    """Side length of world-XY terrain tiles [m]; must divide the total patch dimensions."""
    simulation_capacity_multiplier: int = 2
    """Simulation slots divided by initial particle count, including room for accumulated material."""
    max_cached_particles_per_env: int = 500_000
    """Maximum unloaded particle records per environment. Exhaustion raises before recycling."""
    max_visited_tiles_per_env: int = 4096
    """Maximum visited tile keys per environment, including empty tiles. Exhaustion raises."""

    settle_max_steps: int = 500
    """Maximum startup steps per distinct material template; failure to settle raises."""
    settle_speed: float = 0.03
    """Maximum particle speed [m/s] during the final settling window."""
    settle_quiet_steps: int = 25
    """Consecutive quiet steps required before caching a material template."""
    settle_height_tolerance: float = 0.001
    """Maximum mean-height variation [m] over the final settling window."""
    settle_dt: float = 0.005
    """Offline material-settling timestep [s]."""
    max_material_templates: int = 16
    """Maximum cached settled templates; old templates are evicted when materials change."""
    render_mode: str = "isosurface"
    """Render loaded and saved particle surfaces, or use legacy ``particles`` rendering."""
    surface_resolution: float = 0.02
    """Approximate surface grid spacing [m]; rounded to divide each cache tile."""

    surface_cache_mb: int = 64
    """GPU mesh pool per visible world [MiB]; exhaustion raises instead of dropping terrain history."""
