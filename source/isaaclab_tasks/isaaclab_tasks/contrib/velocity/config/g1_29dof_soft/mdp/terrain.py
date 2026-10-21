# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import trimesh

from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR  # type: ignore


class _TexturedSoftTerrainImporter(TerrainImporter):
    """Add a portable USD texture network alongside the terrain's MDL material."""

    cfg: _TexturedSoftTerrainImporterCfg

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        """Import the terrain and author planar XY texture coordinates."""
        super().import_mesh(name, mesh)
        stage = sim_utils.get_current_stage()
        path = f"{self.cfg.prim_path}/{name}"
        usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{path}/mesh"))
        if self.cfg.disable_collider:
            # The core mesh spawner currently enables collisions regardless of this flag.
            UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim()).CreateCollisionEnabledAttr(False)
        uvs = np.asarray(mesh.vertices[:, :2] * self.cfg.texture_repeat_per_meter, dtype=np.float32)
        UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(Vt.Vec2fArray.FromNumpy(uvs))

        # Keep the MDL output for MDL renderers; Newton reads the universal surface.
        material_path = f"{path}/visualMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        reader = UsdShade.Shader.Define(stage, f"{material_path}/TextureCoordinates")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage, f"{material_path}/ColorTexture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(self.cfg.color_texture))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)


@configclass
class _TexturedSoftTerrainImporterCfg(TerrainImporterCfg):
    """Task-local terrain importer with a PNG-based fallback for Newton viewers."""

    class_type: type[TerrainImporter] = _TexturedSoftTerrainImporter
    color_texture: str = f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K-PNG_Color.png"
    """Diffuse color texture shared with the beach MDL material."""
    texture_repeat_per_meter: tuple[float, float] = (0.25, 0.25)
    """Planar texture repeats along world X and Y [1/m]."""


def _soft_visual_terrain(
    difficulty: float, cfg: terrain_gen.MeshPlaneTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build a closed visual slab while preserving the plane surface and origin.

    Args:
        difficulty: Terrain difficulty in [0, 1].
        cfg: Plane configuration defining tile size and surface height range [m].

    Returns:
        Visual meshes and the tile-center origin on the surface [m], shape [3].
    """
    surface_z = cfg.ground_height_range[0] + difficulty * (cfg.ground_height_range[1] - cfg.ground_height_range[0])
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, surface_z])
    # MuJoCo compiles visual meshes too and rejects the zero-volume plane.
    thickness = 0.05
    transform = np.eye(4)
    transform[:3, 3] = origin - np.array([0.0, 0.0, thickness / 2.0])
    mesh = trimesh.creation.box(extents=(*cfg.size, thickness), transform=transform)
    return [mesh], origin


"""
terrain collections
"""

CurriculumSoftTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),  # size of sub-terrain
        border_width=0.0,
        num_rows=10,
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            # "hard_ground": terrain_gen.MeshPlaneTerrainCfg(
            #     proportion=0.5,
            #     ground_height_range=(0.0, 0.0),
            # ),
            "soft_ground": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.5,
                ground_height_range=(0.0, -0.12),
            ),
        },
    ),
    collision_group=-1,
    # this wont be used in soft terrain
    physics_material=sim_utils.RigidBodyMaterialCfg(
        # friction_combine_mode="average",
        # restitution_combine_mode="average",
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
        albedo_brightness=0.2,
    ),
    max_init_terrain_level=0,
    # debug_vis=True,
)

"""
soft terrain
"""

SoftTerrain = _TexturedSoftTerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        size=(10.0, 10.0),  # size of sub-terrain
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "plane": terrain_gen.MeshPlaneTerrainCfg(
                function=_soft_visual_terrain,
                proportion=1.0,
                ground_height_range=(0.0, 0.0),
            ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        # friction_combine_mode="average",
        # restitution_combine_mode="average",
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        # mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_039/Ground039_4K.mdl", # black sand
        mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K.mdl",  # beach
        project_uvw=True,
        texture_scale=(0.25, 0.25),
        albedo_brightness=0.2,
    ),
    disable_collider=True,
)

RigidPatch = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        size=(50, 50),  # size of sub-terrain
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "plane": terrain_gen.MeshPlaneTerrainCfg(
                proportion=1.0,
                ground_height_range=(0.0005, 0.0005),
            ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        # friction_combine_mode="average",
        # restitution_combine_mode="average",
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.5, 0.5),
        albedo_brightness=0.2,
    ),
)


SoftTerrainVisual = TerrainImporterCfg(
    prim_path="/World/ground_visual",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        # size=(15.0, 15.0),  # size of sub-terrain
        size=(100.0, 100.0),  # size of sub-terrain
        border_width=0.0,
        num_rows=5,
        num_cols=5,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "plane": terrain_gen.MeshPlaneTerrainCfg(
                proportion=1.0,
                ground_height_range=(-0.02, -0.02),
            ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        # friction_combine_mode="average",
        # restitution_combine_mode="average",
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        # mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_039/Ground039_4K.mdl", # black sand
        mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K.mdl",  # beach
        project_uvw=True,
        texture_scale=(0.25, 0.25),
        albedo_brightness=0.2,
    ),
    disable_collider=True,
)

"""
rough terrain
"""
