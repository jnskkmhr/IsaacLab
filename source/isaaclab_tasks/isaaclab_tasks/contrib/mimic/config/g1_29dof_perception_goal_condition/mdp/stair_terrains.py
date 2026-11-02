# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference staircase geometry for mixed imitation/task fine-tuning."""

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.mimic.utils.motion_paired_terrain_generator import load_motion_paired_terrain


def reference_stairs(difficulty: float, cfg: "ReferenceStairsTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Place the unchanged motion terrain in a tile and return its motion-frame origin."""
    with np.load(cfg.motion_files[0], allow_pickle=False) as data:
        terrain = load_motion_paired_terrain(data)
    for path in cfg.motion_files[1:]:
        with np.load(path, allow_pickle=False) as data:
            if load_motion_paired_terrain(data).digest() != terrain.digest():
                raise ValueError("Mixed stair fine-tuning requires motions sharing the same reference terrain.")
    mesh = terrain.mesh()
    translation = np.zeros(3)
    translation[:2] = np.asarray(cfg.size) / 2 - mesh.bounds[:, :2].mean(axis=0)
    mesh.apply_translation(translation)
    meshes = [mesh]
    if terrain.ground_height is not None:
        ground = trimesh.creation.box(extents=(*cfg.size, 0.2))
        ground.apply_translation((*(np.asarray(cfg.size) / 2), terrain.ground_height - 0.1))
        meshes.append(ground)
    return meshes, translation


@configclass
class ReferenceStairsTerrainCfg(SubTerrainBaseCfg):
    """Unmodified geometry shared by the cardinal reference clips."""

    function = reference_stairs
    motion_files: list[str] = MISSING
