# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.scene_cfg import G1SceneCfg as RigidSceneCfg

from ..mdp import soft_flat_terrain


@configclass
class G1SceneCfg(RigidSceneCfg):
    """Retain the mimic robot and sensors above a curriculum of soft-layer depths."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=0.0,
            num_rows=10,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains={
                "soft_ground": terrain_gen.MeshPlaneTerrainCfg(
                    function=soft_flat_terrain,
                    proportion=0.5,
                    ground_height_range=(0.0, -0.12),
                ),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
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
    )
