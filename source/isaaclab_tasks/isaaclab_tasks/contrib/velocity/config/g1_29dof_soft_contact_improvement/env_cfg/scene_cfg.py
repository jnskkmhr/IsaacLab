# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.unitree import UNITREE_G1_29DOF_CFG


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_G1_29DOF_CFG.init_state.replace(pos=(0.0, 0.0, 0.76)),
        # Preserve the reference import settings; the startup event keeps foot margins zero on both terrain types.
        spawn=UNITREE_G1_29DOF_CFG.spawn.replace(collision_props=None),
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
        force_threshold=1.0,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1FourSphereSceneCfg(G1SceneCfg):
    """Reference USD robot with only its four corner spheres enabled on each foot."""

    robot = G1SceneCfg().robot.replace(
        spawn=G1SceneCfg().robot.spawn.replace(
            usd_path=G1SceneCfg().robot.spawn.usd_path.removesuffix(".usda") + ".usd",
            make_uninstanceable=True,  # Allow per-collider overrides on referenced foot geometry.
            collision_props={
                r"/.*_ankle_roll_link/collisions/mesh_0/box": [
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)
                ],
                r"/.*_ankle_roll_link/collisions/mesh_[1-4]/sphere": [
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)
                ],
                r"/.*_ankle_roll_link/collisions/mesh_5/sphere": [
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)
                ],
            },
        ),
    )


@configclass
class G1BoxSceneCfg(G1SceneCfg):
    """Reference USD robot with only its existing box enabled on each foot."""

    robot = G1SceneCfg().robot.replace(
        spawn=G1SceneCfg().robot.spawn.replace(
            usd_path=G1SceneCfg().robot.spawn.usd_path.removesuffix(".usda") + ".usd",
            make_uninstanceable=True,  # Allow per-collider overrides on referenced foot geometry.
            collision_props={
                r"/.*_ankle_roll_link/collisions/mesh_0/box": [
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)
                ],
                r"/.*_ankle_roll_link/collisions/mesh_[1-5]/sphere": [
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)
                ],
            },
        ),
    )
