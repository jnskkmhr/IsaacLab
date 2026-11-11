# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Textured rigid terrain with surface-height queries for resets."""

import trimesh
import warp as wp

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from .terrain_mesh import TexturedTerrainImporter, WarpTerrainMesh


class ContactTerrainImporter(TexturedTerrainImporter):
    """Use the generated surface directly for rigid contact and reset height queries."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        self.background_mesh = WarpTerrainMesh(mesh.vertices, mesh.faces, self.device)
        super().import_mesh(name, mesh)

    def _is_heightfield_collider_requested(self, cfg) -> bool:
        return False

    def configure_env_origins(self, origins=None) -> None:
        super().configure_env_origins(origins)
        self.background_mesh.sample_surface_heights(wp.from_torch(self.env_origins.contiguous(), dtype=wp.vec3))


@configclass
class ContactTerrainImporterCfg(TerrainImporterCfg):
    """Terrain texture settings shared by contact benchmark scenes."""

    class_type: type = ContactTerrainImporter
    mesh_origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color_texture: str = f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K-PNG_Color.png"
    texture_repeat_per_meter: float = 0.125
    color: tuple[float, float, float] = (0.72, 0.55, 0.34)
