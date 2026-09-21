# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain geometry for soft-contact motion tracking."""

import numpy as np
import trimesh

from isaaclab.terrains import MeshPlaneTerrainCfg


def soft_flat_terrain(difficulty: float, cfg: MeshPlaneTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Lower the rigid floor while keeping the reference origin at the soft surface.

    Args:
        difficulty: Terrain difficulty in [0, 1].
        cfg: Plane configuration specifying rigid-floor height range [m].

    Returns:
        Rigid-floor meshes and the motion origin [m] at world Z=0, shape [3].
    """
    floor_z = cfg.ground_height_range[0] + difficulty * (cfg.ground_height_range[1] - cfg.ground_height_range[0])
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    # A solid backing also works for a single tile in MuJoCo, which rejects zero-volume meshes.
    thickness = 0.05
    transform = np.eye(4)
    transform[:3, 3] = (origin[0], origin[1], floor_z - thickness / 2)
    mesh = trimesh.creation.box(extents=(*cfg.size, thickness), transform=transform)
    return [mesh], origin
