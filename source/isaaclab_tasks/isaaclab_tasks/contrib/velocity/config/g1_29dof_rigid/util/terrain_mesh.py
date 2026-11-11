# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Textured terrain and GPU surface-height queries for locomotion."""

import numpy as np
import trimesh
import warp as wp

from pxr import Sdf, UsdGeom, UsdShade, Vt

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter


@wp.func
def surface_height(mesh: wp.uint64, x: float, y: float, ray_z: float, ray_length: float):
    """Cast downward at world XY [m] and return whether the mesh was hit and its height [m]."""
    query = wp.mesh_query_ray(mesh, wp.vec3(x, y, ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
    height = ray_z - query.t
    return query.result, height


@wp.kernel
def sample_heights(mesh: wp.uint64, ray_z: float, ray_length: float, points: wp.array[wp.vec3], missed: wp.array[int]):
    """Replace query-point Z coordinates with terrain heights [m] and count missed rays."""
    i = wp.tid()
    p = points[i]
    hit, height = surface_height(mesh, p[0], p[1], ray_z, ray_length)
    if hit:
        points[i] = wp.vec3(p[0], p[1], height)
    else:
        wp.atomic_add(missed, 0, 1)


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


class WarpTerrainMesh:
    """Wrap terrain vertices and faces in a Warp mesh for GPU height queries.

    The mesh provides accelerated ray queries for terrain-relative robot resets.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        self.bounds = np.stack((self.vertices.min(axis=0), self.vertices.max(axis=0)))
        self.mesh = wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3, device=device),
            indices=wp.array(self.faces.flatten(), dtype=int, device=device),
        )
        self.height_query_start_z = float(self.bounds[1, 2] + 1.0)
        self.height_query_max_distance = float(self.bounds[1, 2] - self.bounds[0, 2] + 2.0)
        self.env_origins = None
        """Initial environment reset positions on the background terrain, in world coordinates [m]."""
        self.env_clone_origins = None
        """Environment clone-grid positions, in world coordinates [m]."""

    def sample_surface_heights(self, query_positions: wp.array) -> None:
        """Replace world-space query points' Z coordinates by surface heights [m]."""
        terrain_query_miss_count = wp.zeros(1, dtype=int, device=self.mesh.device)
        wp.launch(
            sample_heights,
            dim=len(query_positions),
            inputs=[
                self.mesh.id,  # mesh
                self.height_query_start_z,  # ray_z
                self.height_query_max_distance,  # ray_length
            ],
            outputs=[query_positions, terrain_query_miss_count],
            device=self.mesh.device,
        )
        if terrain_query_miss_count.numpy()[0]:
            raise ValueError("Shared terrain height query missed the surface; check terrain coverage and spawn spacing")
