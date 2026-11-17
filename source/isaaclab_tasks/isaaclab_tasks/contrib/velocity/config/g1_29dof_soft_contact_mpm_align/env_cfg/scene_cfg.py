# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import LegacyRayCaster

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact.mdp.rough_terrain import CurriculumSoftRoughTerrainCfg
from isaaclab_tasks.contrib.velocity.config.vel_mdp.rough_height import elevated_grid_pattern

from isaaclab_assets.robots.unitree import UNITREE_G1_29DOF_BOX_FOOT_CFG

from .terrain_cfg import ROUGH_TERRAINS_CFG


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_G1_29DOF_BOX_FOOT_CFG.init_state.replace(pos=(0.0, 0.0, 0.76)),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        max_init_terrain_level=0,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=10,
            num_cols=10,
            border_width=0.0,
            curriculum=True,
            use_cache=False,
            sub_terrains={
                "hard_ground": MeshPlaneTerrainCfg(size=(8.0, 8.0), proportion=0.5, ground_height_range=(0.0, 0.0)),
                "soft_ground": MeshPlaneTerrainCfg(size=(8.0, 8.0), proportion=0.5, ground_height_range=(0.0, -0.12)),
            },
        ),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.005,
        force_threshold=5.0,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class HeightGridPatternCfg(patterns.GridPatternCfg):
    """Start downward rays above the robot in world Z [m]."""

    func = elevated_grid_pattern
    height: float = 20.0


@configclass
class G1RoughSceneCfg(G1SceneCfg):
    """Box-foot robot with surface-relative measurements on generated rough terrain."""

    terrain = CurriculumSoftRoughTerrainCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        max_init_terrain_level=0,
        terrain_generator=ROUGH_TERRAINS_CFG.copy(),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    height_scanner = RayCasterCfg(
        class_type=LegacyRayCaster,
        prim_path="{ENV_REGEX_NS}/Robot/.*pelvis",
        ray_alignment="yaw",
        pattern_cfg=HeightGridPatternCfg(resolution=0.1, size=(0.2, 0.2)),
        mesh_prim_paths=["/World/ground/terrain"],
        global_world_only=True,
    )
    left_foot_scanner = height_scanner.replace(
        prim_path="{ENV_REGEX_NS}/Robot/.*left_ankle_roll_link",
        pattern_cfg=HeightGridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
    )
    right_foot_scanner = left_foot_scanner.replace(prim_path="{ENV_REGEX_NS}/Robot/.*right_ankle_roll_link")
