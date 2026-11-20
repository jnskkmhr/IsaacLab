# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Textured scene terrain, the sand layer settings, and the support geometry under the sand."""

from __future__ import annotations

import math
from dataclasses import MISSING

import numpy as np
import trimesh
import warp as wp
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

from pxr import Sdf, UsdGeom, UsdShade, Vt

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


@wp.kernel
def sample_heights(mesh: wp.uint64, ray_z: float, ray_length: float, points: wp.array[wp.vec3], missed: wp.array[int]):
    """Replace query-point Z coordinates with terrain heights [m] and count missed rays."""
    i = wp.tid()
    p = points[i]
    query = wp.mesh_query_ray(mesh, wp.vec3(p[0], p[1], ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
    if query.result:
        points[i] = wp.vec3(p[0], p[1], ray_z - query.t)
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


@configclass
class SandLayerCfg:
    """Depth, sampling and material of the sand layer on top of the terrain.

    The moving window itself is configured on the MPM solver's ``StreamingCfg``.
    """

    particle_depth: float = MISSING
    """Vertical particle-layer depth below the undisturbed terrain surface [m]."""
    robot_spawn_height: float = MISSING
    """Initial robot root height above the sand surface [m]."""
    voxel_size: float = MISSING
    """MPM background-grid voxel size [m]."""
    particles_per_cell: float = MISSING
    """Particle samples along each voxel dimension (IsaacLab MPMGridCfg convention)."""
    jitter: float = MISSING
    """Total width of the initial particle-position jitter interval [m].

    Each X, Y, and Z coordinate receives an independent uniform offset in
    ``[-jitter / 2, jitter / 2]``, matching :class:`MPMGridCfg`.
    """
    material: MPMParticleMaterialCfg = MISSING
    """Particle density and constitutive parameters, shared by initial and recycled particles."""
    floor_thickness: float = MISSING
    """Thickness of the static supporting floor below the sand [m]."""
    show_boundary_particles: bool = False
    """Render the kinematic ring along with simulated particles."""
    render_mode: str = "isosurface"
    """With a tile cache, draw loaded sand as a surface, or as ``particles``. Without a cache, particles are drawn."""
    surface_resolution: float = 0.015
    """Approximate surface grid spacing [m]; rounded to divide each cache tile."""
    surface_smoothing: float = 1.5
    """Multiplier of the particle spacing used to reconstruct the surface; at least 1."""
    surface_cache_mb: int = 64
    """Memory for the baked surface meshes of tiles that left the pool [MiB]; zero shows the plain terrain there."""
    visual_color: tuple[float, float, float] = (0.72, 0.55, 0.34)
    """Particle-emitter RGB color; viewer particle_color controls rendered particles."""

    def validate_geometry(self) -> None:
        """Reject invalid geometry before creating the scene."""
        positive_dimensions = (self.particle_depth, self.voxel_size, self.particles_per_cell, self.floor_thickness)
        if not all(math.isfinite(dimension) and dimension > 0 for dimension in positive_dimensions):
            raise ValueError("Sand depth, voxel size, sampling and floor thickness must be positive and finite")
        if not math.isfinite(self.jitter) or self.jitter < 0.0:
            raise ValueError("Particle jitter width must be finite and non-negative [m]")


class WarpTerrainMesh:
    """Wrap terrain vertices and faces in a Warp mesh for GPU height queries.

    The mesh provides accelerated ray queries; bounds and environment origins are
    used for particle placement. Particle rendering reads simulation positions.
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
        """Environment clone-grid positions used to author the initial particles, in world coordinates [m]."""

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


class MPMTerrainImporter(TexturedTerrainImporter):
    """Use one IsaacLab terrain with a global support collider for each subsolver."""

    def __init__(self, cfg: MPMTerrainImporterCfg):
        cfg.validate_geometry()
        super().__init__(cfg)

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        """Keep the generated surface for height queries and lower the supporting collider."""
        self.background_mesh = WarpTerrainMesh(mesh.vertices, mesh.faces, self.device)
        # Preserve the generated surface; only the supporting collider is lowered.
        support_mesh = mesh.copy()
        support_mesh.apply_translation((0, 0, -self.cfg.sand.particle_depth))

        # physics_cfg.py assigns "terrain" to MJWarp and "mpm_ground_mesh" to MPM.
        # The coupler does not allow one collider to belong to both solvers.
        super().import_mesh(name, support_mesh)
        super().import_mesh("mpm_ground", support_mesh)
        # Both solvers need a collider, but only one copy of the surface should be drawn.
        UsdGeom.Imageable(
            sim_utils.get_current_stage().GetPrimAtPath(f"{self.cfg.prim_path}/mpm_ground")
        ).MakeInvisible()

    def _is_heightfield_collider_requested(self, cfg):
        # The installed implicit MPM collision path consumes triangle meshes.
        return False

    def configure_env_origins(self, origins=None):
        """Place robots on grid or terrain-tile origins and sample spawn heights [m]."""
        super().configure_env_origins(origins)
        # Particles are authored at clone-grid origins, independently of reset tile selection.
        env_clone_origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)
        self.background_mesh.env_clone_origins = env_clone_origins.cpu().numpy().copy()
        env_origin_query_positions = wp.from_torch(self.env_origins.contiguous(), dtype=wp.vec3)
        self.background_mesh.sample_surface_heights(env_origin_query_positions)
        self.background_mesh.env_origins = self.env_origins.detach().cpu().numpy().copy()


@configclass
class MPMTerrainImporterCfg(TexturedTerrainImporterCfg):
    """Standard generated-terrain settings plus the sand layer resting on it."""

    class_type: type = MPMTerrainImporter
    use_terrain_origins: bool = False

    sand: SandLayerCfg = MISSING
    """Depth, sampling and material of the sand layer."""

    def validate_geometry(self) -> None:
        """Validate the sand layer and require a generated terrain mesh."""
        self.sand.validate_geometry()
        if self.terrain_type != "generator" or self.terrain_generator is None:
            raise ValueError("A sand layer requires terrain_type='generator'")
        terrain_generator_cfg = self.terrain_generator
        if len(terrain_generator_cfg.size) != 2 or not all(
            math.isfinite(dimension) and dimension > 0 for dimension in terrain_generator_cfg.size
        ):
            raise ValueError("terrain_generator.size must contain two positive finite values")
        if terrain_generator_cfg.num_rows < 1 or terrain_generator_cfg.num_cols < 1:
            raise ValueError("Terrain generator row and column counts must be positive")
