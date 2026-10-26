# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

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
from isaaclab_assets.robots.unitree import UNITREE_G1_29DOF_CFG

from ..mdp.actuators import DelayedImplicitActuatorCfg
from . import terrain_cfg

G1_29DOF_BeyondMimic_delay = UNITREE_G1_29DOF_CFG.copy()
G1_29DOF_BeyondMimic_delay.spawn = sim_utils.UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
    # Disable hands to accelerate training.
    variants={"Physics": "PhysX", "left_hand": "None", "right_hand": "None"},
    activate_contact_sensors=True,
    rigid_props=PhysxRigidBodyCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=PhysxArticulationCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    ),
)
G1_29DOF_BeyondMimic_delay.init_state.pos = (0.0, 0.0, 0.76)
G1_29DOF_BeyondMimic_delay.actuators = {
    name: DelayedImplicitActuatorCfg(
        joint_names_expr=cfg.joint_names_expr,
        joint_effort_limit=cfg.effort_limit_sim,
        joint_velocity_limit=cfg.velocity_limit_sim,
        stiffness=cfg.stiffness,
        damping=cfg.damping,
        armature=cfg.armature,
        min_delay=0,
        max_delay=4,
    )
    for name, cfg in G1_29DOF_BeyondMimic_delay.actuators.items()
}


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
    robot: ArticulationCfg = G1_29DOF_BeyondMimic_delay.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

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
    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
        global_world_only=True,
    )
    smooth_interval_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        update_period=0.0,
        history_length=5,
        track_air_time=False,
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
