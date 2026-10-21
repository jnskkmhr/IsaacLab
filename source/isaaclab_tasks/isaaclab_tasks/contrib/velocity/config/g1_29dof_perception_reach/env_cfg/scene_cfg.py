# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 scene with stairs, terrain landing targets, and a torso-mounted elevation scanner."""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG

from .terrain_cfg import BorderedRandomGridTerrainCfg


@configclass
class G1PerceptionSceneCfg(InteractiveSceneCfg):
    """Robot, terrain, and sensor settings owned by the perception task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            # border_width=5.0,
            border_width=20.0,
            num_rows=10,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            # Use the +X flat border as a landing strip, clear of the terrain interior.
            sub_terrains={
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(
                            num_patches=64,
                            # Check a 0.3 m standing radius and sample the center last for goal Z.
                            patch_radius=[0.15, 0.3, 0.0],
                            max_height_diff=0.02,
                            x_range=(3.35, 3.5),
                            y_range=(-0.5, 0.5),
                        ),
                    },
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(
                            num_patches=64,
                            patch_radius=[0.15, 0.3, 0.0],
                            max_height_diff=0.02,
                            x_range=(3.35, 3.5),
                            y_range=(-0.5, 0.5),
                        ),
                    },
                ),
                "boxes": BorderedRandomGridTerrainCfg(
                    proportion=0.2,
                    grid_width=0.45,
                    grid_height_range=(0.05, 0.2),
                    platform_width=3.0,
                    border_width=1.0,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(
                            num_patches=64,
                            patch_radius=[0.15, 0.3, 0.0],
                            max_height_diff=0.02,
                            x_range=(3.35, 3.5),
                            y_range=(-0.5, 0.5),
                        ),
                    },
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(0.02, 0.10),
                    noise_step=0.02,
                    border_width=1.0,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(
                            num_patches=64,
                            patch_radius=[0.15, 0.3, 0.0],
                            max_height_diff=0.02,
                            x_range=(3.35, 3.5),
                            y_range=(-0.5, 0.5),
                        ),
                    },
                ),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.1,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=1.0,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(
                            num_patches=64,
                            patch_radius=[0.15, 0.3, 0.0],
                            max_height_diff=0.02,
                            x_range=(3.35, 3.5),
                            y_range=(-0.5, 0.5),
                        ),
                    },
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.1,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=1.0,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(
                            num_patches=64,
                            patch_radius=[0.15, 0.3, 0.0],
                            max_height_diff=0.02,
                            x_range=(3.35, 3.5),
                            y_range=(-0.5, 0.5),
                        ),
                    },
                ),
            },
        ),
        max_init_terrain_level=0,
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
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0), ordering="yx"),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=True,
    )
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(
    #         func=elevation_grid_pattern, resolution=0.1, size=(3.2, 2.0), ordering="yx"
    #     ),
    #     mesh_prim_paths=["/World/ground"],
    #     global_world_only=True,
    #     debug_vis=False,
    # )
    # Sole center is 0.038 m forward of the ankle-roll frame for the box-foot asset.
    left_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.038, 0.0, 1.0)),
        ray_alignment="yaw",
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(0.24, 0.12), ordering="yx"),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(0.065 + 0.141, 0.0368 * 2), ordering="yx"),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=True,
    )
    right_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.038, 0.0, 1.0)),
        ray_alignment="yaw",
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(0.24, 0.12), ordering="yx"),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(0.065 + 0.141, 0.0368 * 2), ordering="yx"),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=True,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
