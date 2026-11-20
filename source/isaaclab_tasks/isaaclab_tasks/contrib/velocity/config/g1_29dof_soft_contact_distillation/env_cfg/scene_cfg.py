# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        max_init_terrain_level=0,
        physics_material=PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=10,
            num_cols=10,
            border_width=0.0,
            curriculum=True,
            use_cache=False,
            sub_terrains={
                "hard_ground": MeshPlaneTerrainCfg(proportion=0.5, ground_height_range=(0.0, 0.0)),
                "soft_ground": MeshPlaneTerrainCfg(proportion=0.5, ground_height_range=(0.0, -0.12)),
            },
        ),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.005,
        force_threshold=1.0,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
