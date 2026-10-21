# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Box terrain with a flat landing border for the reach task."""

import numpy as np
import trimesh

from isaaclab.terrains import MeshRandomGridTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains import random_grid_terrain
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils.configclass import configclass


@configclass
class BorderedRandomGridTerrainCfg(MeshRandomGridTerrainCfg):
    """Keep the random boxes inside a flat border with surface height Z = 0."""

    function: str = "{DIR}.terrain_cfg:random_grid_with_border"
    border_width: float = 1.0
    """Width of the flat border on each side of the terrain tile, in meters."""


def random_grid_with_border(
    difficulty: float, cfg: BorderedRandomGridTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate the standard box grid in the interior and surround it with a flat landing area."""
    if not 0.0 < cfg.border_width < min(cfg.size) / 2.0:
        raise ValueError("The box terrain border must be positive and less than half the tile size.")
    inner_size = (cfg.size[0] - 2.0 * cfg.border_width, cfg.size[1] - 2.0 * cfg.border_width)
    meshes, origin = random_grid_terrain(difficulty, cfg.replace(size=inner_size))
    offset = np.array([cfg.border_width, cfg.border_width, 0.0])
    for mesh in meshes:
        mesh.apply_translation(offset)
    meshes += make_border(
        size=cfg.size,
        inner_size=inner_size,
        height=1.0,
        position=(cfg.size[0] / 2.0, cfg.size[1] / 2.0, -0.5),
    )
    return meshes, origin + offset
