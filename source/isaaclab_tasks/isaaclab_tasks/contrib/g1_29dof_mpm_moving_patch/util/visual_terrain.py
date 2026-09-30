# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Textured scene terrain, independent of MPM particle and support-collider handling."""

import numpy as np
import trimesh

from pxr import Sdf, UsdGeom, UsdShade, Vt

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


class TexturedTerrainImporter(TerrainImporter):
    """Add UV coordinates and a USD preview texture alongside the configured MDL material."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        """Import scene geometry with planar UVs and an MDL-compatible preview material."""
        if any(self.cfg.mesh_origin_offset):
            mesh = mesh.copy()
            mesh.apply_translation(self.cfg.mesh_origin_offset)
        super().import_mesh(name, mesh)
        if self.cfg.disable_visual:
            return
        stage = sim_utils.get_current_stage()
        path = f"{self.cfg.prim_path}/{name}"
        usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{path}/mesh"))
        cfg = self.cfg
        uvs = np.asarray((mesh.vertices[:, :2] - mesh.bounds[0, :2]) * cfg.texture_repeat_per_meter, dtype=np.float32)
        UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(Vt.Vec2fArray.FromNumpy(uvs))

        # Preserve MDL for RTX and provide the standard USD texture network for GL.
        material_path = f"{path}/visualMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        preview = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        preview.CreateIdAttr("UsdPreviewSurface")
        diffuse = preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if cfg.color_texture:
            reader = UsdShade.Shader.Define(stage, f"{material_path}/TextureCoordinates")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            texture = UsdShade.Shader.Define(stage, f"{material_path}/ColorTexture")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(cfg.color_texture))
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            diffuse.ConnectToSource(texture.ConnectableAPI(), "rgb")
        else:
            diffuse.Set(cfg.color)
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)


@configclass
class TexturedTerrainImporterCfg(TerrainImporterCfg):
    """Standard terrain import with a portable color texture for GL and RTX."""

    class_type: type = TexturedTerrainImporter

    mesh_origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation of imported mesh vertices along the terrain's local axes [m].

    Does not shift environment origins or the source mesh. Only applies to mesh imports.
    """
    visual_material: sim_utils.MdlFileCfg = sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K.mdl",
        project_uvw=False,
        texture_scale=(1.0, 1.0),
    )
    color_texture: str | None = f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K-PNG_Color.png"
    """Color texture for the USD preview material; None uses color."""
    texture_repeat_per_meter: float = 0.125
    """Number of texture repeats per world-space distance [1/m]."""
    color: tuple[float, float, float] = (0.72, 0.55, 0.34)
    """Untextured preview material RGB color."""
