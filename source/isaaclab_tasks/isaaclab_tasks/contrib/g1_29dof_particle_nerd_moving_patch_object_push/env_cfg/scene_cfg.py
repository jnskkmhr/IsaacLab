# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 assets, geometry and material configuration for recycling terrain."""

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import UNITREE_G1_29DOF_WITH_HAND_CFG

from . import terrain_cfg
from .terrain_cfg import BackgroundTerrainImporterCfg, MovingPatchTerrainCfg

VOXEL_SIZE = 0.04
"""Particle sampling voxel size [m]."""
SAND_DEPTH = 0.3
"""Additional depth of the sand layer above the ground [m]."""
FOOT_CONTACT_MARGIN = VOXEL_SIZE * 0.5
"""Sole contact margin [m], matching the validated fixed-terrain example."""


ROBOT_CFG = UNITREE_G1_29DOF_WITH_HAND_CFG.copy()
ROBOT_CFG.spawn.collision_props = {
    r"/.*_ankle_roll_link/.*": [NewtonCollisionCfg(contact_margin=FOOT_CONTACT_MARGIN, contact_gap=0.0)]
}
LOCOMOTION_REFERENCE_JOINT_POS = dict(ROBOT_CFG.init_state.joint_pos)
MANIPULATION_REFERENCE_JOINT_POS = {
    "waist_.*_joint": 0.0,
    ".*_shoulder_pitch_joint": -0.42,
    "left_shoulder_roll_joint": 0.17,
    "right_shoulder_roll_joint": -0.17,
    "left_shoulder_yaw_joint": 0.13,
    "right_shoulder_yaw_joint": -0.13,
    ".*_elbow_joint": 1.0,
    "left_wrist_roll_joint": -0.15,
    "right_wrist_roll_joint": 0.15,
    ".*_wrist_pitch_joint": -0.56,
    "left_wrist_yaw_joint": -0.15,
    "right_wrist_yaw_joint": 0.15,
}
ROBOT_CFG.init_state.joint_pos = {
    expression: value
    for expression, value in LOCOMOTION_REFERENCE_JOINT_POS.items()
    if not any(part in expression for part in ("waist", "shoulder", "elbow", "wrist"))
} | MANIPULATION_REFERENCE_JOINT_POS
# Nominal PD gains: legs/waist from locomotion ltn8eu4n; arms from manipulation 6pbvl2df.
for actuator_name, gains in {
    "feet": {"stiffness": 28.50124619574858, "damping": 1.814445686584846},
    "legs": {
        "stiffness": {
            ".*_knee_joint": 99.09842777666113,
            ".*_hip_yaw_joint": 40.17923847137318,
            ".*_hip_roll_joint": 99.09842777666113,
            ".*_hip_pitch_joint": 40.17923847137318,
        },
        "damping": {
            ".*_knee_joint": 6.3088018534966395,
            ".*_hip_yaw_joint": 2.5578897650279457,
            ".*_hip_roll_joint": 6.3088018534966395,
            ".*_hip_pitch_joint": 2.5578897650279457,
        },
    },
    "waist": {"stiffness": 28.50124619574858, "damping": 1.814445686584846},
    "waist_yaw": {"stiffness": 40.17923847137318, "damping": 2.5578897650279457},
    "arms": {
        "stiffness": {
            ".*_elbow_joint": 14.25062309787429,
            ".*_wrist_yaw_joint": 33.55665496178558,
            ".*_wrist_roll_joint": 28.50124619574858,
            ".*_wrist_pitch_joint": 33.55665496178558,
            ".*_shoulder_yaw_joint": 14.25062309787429,
            ".*_shoulder_roll_joint": 14.25062309787429,
            ".*_shoulder_pitch_joint": 14.25062309787429,
        },
        "damping": {
            ".*_elbow_joint": 0.907222843292423,
            ".*_wrist_yaw_joint": 2.13628300438,
            ".*_wrist_roll_joint": 1.814445686584846,
            ".*_wrist_pitch_joint": 2.13628300438,
            ".*_shoulder_yaw_joint": 0.907222843292423,
            ".*_shoulder_roll_joint": 0.907222843292423,
            ".*_shoulder_pitch_joint": 0.907222843292423,
        },
    },
}.items():
    ROBOT_CFG.actuators[actuator_name].stiffness = gains["stiffness"]
    ROBOT_CFG.actuators[actuator_name].damping = gains["damping"]


@configclass
class SceneCfg(InteractiveSceneCfg):
    """G1, shared terrain, particles and lights for the moving-patch task."""

    terrain: BackgroundTerrainImporterCfg = BackgroundTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        terrain_generator=terrain_cfg.FLAT_TERRAINS_CFG.copy(),
        max_init_terrain_level=0,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
        moving_patch_terrain=MovingPatchTerrainCfg(
            simulated_terrain_size=(3.4, 2.4),
            boundary_terrain_size=0.2,
            tracked_body="pelvis",
            robot_spawn_height=0.76,
            patch_discretization_step=0.2,
            particle_depth=SAND_DEPTH,
            voxel_size=VOXEL_SIZE,
            particles_per_cell=1.25,
            jitter=0.05,
            material=MPMParticleMaterialCfg(density=1600.0, young_modulus=1e15, friction=0.48, yield_pressure=1e15),
            floor_thickness=0.1,
            show_boundary_particles=False,
            visual_color=(0.72, 0.55, 0.34),
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.0, 0.335)),
        spawn=sim_utils.CuboidCfg(
            size=(0.65, 0.65, 0.65),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=UsdPhysicsRigidBodyCfg(),
            collision_props=[UsdPhysicsCollisionCfg(), NewtonCollisionCfg(contact_margin=0.005, contact_gap=0.0)],
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.75)),
        ),
    )

    height_scanner = RayCasterCfg(
        # the URDF importer nests bodies along the kinematic tree, so match the leaf name anywhere
        prim_path="{ENV_REGEX_NS}/Robot/.*pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2)),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )

    sand: MPMObjectCfg | None = None

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self):
        """Initialize particles when this scene config is constructed on its own."""
        if self.sand is None:
            self.configure_terrain()

    def configure_terrain(self) -> None:
        """Derive all authored geometry from the final terrain configuration."""
        self.terrain.validate_geometry()

        terrain = self.terrain.moving_patch_terrain
        x, y = terrain.total_patch_size
        # TerrainImporter supplies each robot's world-space spawn origin and height.
        self.robot.init_state.pos = (0.0, 0.0, terrain.robot_spawn_height)
        self.sand = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            init_state=MPMObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            spawn=MPMGridCfg(
                lower=(-x / 2, -y / 2, -terrain.particle_depth),
                upper=(x / 2, y / 2, 0.0),
                voxel_size=terrain.voxel_size,
                particles_per_cell=terrain.particles_per_cell,
                particle_placement="cell_center",
                jitter=terrain.jitter,
                material=terrain.material,
                visual_color=terrain.visual_color,
            ),
        )
