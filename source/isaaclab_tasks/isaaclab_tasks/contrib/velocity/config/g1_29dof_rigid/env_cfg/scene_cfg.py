# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

##
# Pre-defined configs
##
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR, UNITREE_G1_29DOF_BOX_FOOT_CFG

from ..util.terrain import ContactTerrainImporterCfg
from . import terrain_cfg


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        terrain_generator=terrain_cfg.ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
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
    # robots
    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    # sensors
    height_scanner = RayCasterCfg(
        # the URDF importer nests bodies along the kinematic tree, so match the leaf name anywhere
        # prim_path="{ENV_REGEX_NS}/Robot/.*pelvis",
        prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2)),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=True,
    )
    # multi-body contact reporting
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1RoughSceneCfg(InteractiveSceneCfg):
    """Box-foot G1, rough terrain, height scanner, and foot contact sensors."""

    terrain = ContactTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        terrain_generator=terrain_cfg.CONTACT_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K.mdl",
            project_uvw=False,
            texture_scale=(1.0, 1.0),
        ),
    )
    robot = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_G1_29DOF_BOX_FOOT_CFG.init_state.replace(pos=(0.0, 0.0, 0.76)),
        spawn=UNITREE_G1_29DOF_BOX_FOOT_CFG.spawn.replace(collision_props=None),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2)),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        history_length=1,
        update_period=0.005,
        track_air_time=True,
        force_threshold=5.0,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
