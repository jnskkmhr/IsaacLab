# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough RFT surfaces with a curriculum on buried rigid support depth."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.util.terrain import (
    ContactTerrainImporter,
    ContactTerrainImporterCfg,
)


class CurriculumSoftRoughTerrain(ContactTerrainImporter):
    """Keep surface queries at the rough surface and lower rigid support in alternate columns.

    Even columns remain rigid at the surface. Odd columns have the same rough shape
    translated downward by the row's support depth. Curriculum progression moves robots
    between rows; no collider is moved or rebuilt while simulation is running.
    """

    cfg: CurriculumSoftRoughTerrainCfg

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        generator = self.cfg.terrain_generator
        if generator.num_cols % 2:
            raise ValueError("Hybrid rough terrain requires an even number of columns for rigid/soft pairs.")
        low, high = self.cfg.support_depth_range
        if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low <= high):
            raise ValueError("support_depth_range must contain finite depths with 0 <= minimum <= maximum.")
        if not self.cfg.disable_collider:
            raise ValueError("The RFT surface must disable collisions; rigid support is authored separately.")
        if any(self.cfg.mesh_origin_offset):
            raise ValueError("Hybrid rough terrain requires mesh_origin_offset=(0, 0, 0).")

        # This is the surface used by RFT, the scanner, and robot reset height queries.
        super().import_mesh(name, mesh)

        # Classify triangles by tile, keeping the surrounding border at its original height.
        centers = mesh.triangles_center[:, :2]
        tile = np.floor(centers / np.asarray(generator.size) + [generator.num_rows / 2, generator.num_cols / 2])
        rows, cols = tile.astype(np.int64).T
        inside = (rows >= 0) & (rows < generator.num_rows) & (cols >= 0) & (cols < generator.num_cols)
        soft = inside & (cols % 2 == 1)
        depths = np.linspace(low, high, generator.num_rows)
        self.support_depths = np.zeros((generator.num_rows, generator.num_cols))
        self.support_depths[:, 1::2] = depths[:, None]
        face_depth = np.zeros(len(mesh.faces))
        face_depth[soft] = depths[rows[soft]]
        support_parts = []
        for depth in np.unique(face_depth):
            part = mesh.submesh([np.flatnonzero(face_depth == depth)], append=True, repair=False)
            part.apply_translation((0.0, 0.0, -float(depth)))
            support_parts.append(part)
        support_mesh = trimesh.util.concatenate(support_parts)
        # Keep the support inside the terrain root imported by the physics backend.
        # RFT queries background_mesh; the height scanner targets only the surface prim.
        create_prim_from_mesh(
            self.cfg.prim_path + "/support",
            support_mesh,
            physics_material=self.cfg.physics_material,
            disable_visual=True,
        )


@configclass
class CurriculumSoftRoughTerrainCfg(ContactTerrainImporterCfg):
    """Rough surface and rigid support settings for mixed rigid/RFT training."""

    class_type: type = CurriculumSoftRoughTerrain
    disable_collider: bool = True
    support_depth_range: tuple[float, float] = (0.0, 0.12)
    """Rigid support depth below the rough surface at the first and last row [m]."""
