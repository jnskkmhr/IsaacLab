# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Flat rigid surfaces for the particle-free half of a mixed training batch."""

import trimesh

from isaaclab.terrains import TerrainImporter

from ..g1_29dof_mpm_moving_patch.util.terrain import BackgroundTerrainImporter


class MixedTerrainImporter(BackgroundTerrainImporter):
    """Keep the MPM support floor and add rigid surfaces at the initial sand height.

    The first half of clone-grid environments contains sand. The second half has
    one rigid tile per environment, with its upper surface at world Z = 0.
    """

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        super().import_mesh(name, mesh)
        origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing).cpu().numpy()
        tiles = []
        thickness = self.cfg.moving_patch_terrain.floor_thickness
        for origin in origins[self.cfg.num_envs // 2 :]:
            tile = trimesh.creation.box(extents=(self.cfg.env_spacing, self.cfg.env_spacing, thickness))
            tile.apply_translation((origin[0], origin[1], -thickness / 2))
            tiles.append(tile)
        # Unlike the buried support floor, the rigid group's walking surface is visible.
        cfg = self.cfg
        self.cfg = cfg.replace(disable_visual=False)
        try:
            TerrainImporter.import_mesh(self, "rigid_surface", trimesh.util.concatenate(tiles))
        finally:
            self.cfg = cfg
