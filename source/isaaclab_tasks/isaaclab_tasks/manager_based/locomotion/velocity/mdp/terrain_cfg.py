# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR # type: ignore

"""
base flat terrain.
"""

CurriculumSoftTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0), # size of sub-terrain
        border_width=0.0,
        num_rows=10,
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "hard_ground": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.2, 
                ground_height_range=(0.0, 0.0),
                ),
            "soft_ground": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.8, 
                ground_height_range=(0.0, -0.12),
                ),
        },
    ),
    collision_group=-1,
    # this wont be used in soft terrain 
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=0.5,
        dynamic_friction=0.5,
    ),
    visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
            albedo_brightness=0.5,
        ),
    max_init_terrain_level=0,
)

SoftTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(100.0, 100.0), # size of sub-terrain
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
                ground_height_range=(0.0, 0.0),
                ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=0.5,
        dynamic_friction=0.5,
    ),
    visual_material=sim_utils.MdlFileCfg(
            # mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            # mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_039/Ground039_4K.mdl", # black sand
            mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K.mdl", # beach
            project_uvw=True,
            texture_scale=(0.25, 0.25),
            albedo_brightness=0.2,
        ),
)

RigidPatch = TerrainImporterCfg(
    prim_path="/World/rigid_patch",
    terrain_type="generator",
    terrain_generator= terrain_gen.TerrainGeneratorCfg(
        size=(1.5, 1.5), # size of sub-terrain
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
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=0.5,
        dynamic_friction=0.5,
    ),
    visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            # mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_039/Ground039_4K.mdl", # black sand
            # mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K.mdl", # beach
            project_uvw=True,
            texture_scale=(0.5, 0.5),
            albedo_brightness=0.2,
        ),
)