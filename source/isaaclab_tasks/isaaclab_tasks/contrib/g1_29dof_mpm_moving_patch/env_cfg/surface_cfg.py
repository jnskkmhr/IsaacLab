# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Display settings; these do not alter the MPM discretization or particle state."""

from isaaclab.utils.configclass import configclass


@configclass
class SandSurfaceCfg:
    show_surface: bool = True
    show_particles: bool = False
    resolution: float = 0.025
    """Surface grid spacing [m], independent of the physics voxel size."""
    tile_size: float = 0.5
    """XY region size [m] for live/static surface classification."""
    kernel_radius: float = 0.048
    """Newton density reconstruction kernel radius [m]."""
    density_threshold: float = 0.5
    max_grid_cells: int = 1_000_000
    """Per-environment sparse reconstruction capacity."""
    max_vertices: int = 262_144
    max_triangles: int = 524_288
    """Per-mesh allocation capacities; the viewer receives populated ranges only."""
    update_interval: int = 1
    """Update on each rendered physics step by default."""
