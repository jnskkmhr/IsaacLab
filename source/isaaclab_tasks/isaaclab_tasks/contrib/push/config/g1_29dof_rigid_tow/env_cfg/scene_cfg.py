# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1, a sled on a rope, and flat rigid ground of the tow task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.util.terrain import ContactTerrainImporterCfg

from isaaclab_assets.robots.unitree import UNITREE_G1_29DOF_BOX_FOOT_CFG

SLED_ASSET = "sled"
"""Scene name of the towed object."""
SLED_NAME = "Sled"
"""Prim and body name of the towed object."""
SLED_SIZE = (0.3, 0.3, 0.3)
"""Length along the tow direction, width and height of the sled [m]; as in the sand task."""
SLED_MASS = 15.0
"""Mass of the sled [kg]; as in the sand task."""
SLED_FRICTION = 0.5
"""Friction coefficient of the sled surface [-]; as in the sand task."""
SLED_RESET_DISTANCE = 1.15
"""Distance of the sled center behind the pelvis at reset [m]; the rope is just slack there."""
GROUND_FRICTION = 0.5
"""Friction coefficient of the ground [-].

Contact friction is the larger of the two shapes' coefficients, so the ground must not exceed the sled's
value for the sled to slide at ``SLED_FRICTION``. The robot's feet keep their own, higher coefficient.
"""


@configclass
class G1TowSceneCfg(InteractiveSceneCfg):
    """G1 on flat rigid ground with a sled resting behind it."""

    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_G1_29DOF_BOX_FOOT_CFG.init_state.replace(pos=(0.0, 0.0, 0.76)),
    )
    terrain = ContactTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        max_init_terrain_level=0,
        physics_material=RigidBodyMaterialCfg(static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION),
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=10,
            num_cols=10,
            border_width=0.0,
            curriculum=False,
            use_cache=False,
            sub_terrains={
                "hard_ground": MeshPlaneTerrainCfg(size=(8.0, 8.0), proportion=1.0, ground_height_range=(0.0, 0.0)),
            },
        ),
    )
    sled: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + SLED_NAME,
        spawn=sim_utils.CuboidCfg(
            size=SLED_SIZE,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=SLED_MASS),
            collision_props=UsdPhysicsCollisionCfg(),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=SLED_FRICTION, dynamic_friction=SLED_FRICTION),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.75), roughness=0.6),
        ),
        # Behind the robot, just above the ground; the reset event places it relative to the robot.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-SLED_RESET_DISTANCE, 0.0, 0.5 * SLED_SIZE[2] + 0.02)),
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
