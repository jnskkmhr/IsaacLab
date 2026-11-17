# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot, dynamic box, and flat sand configuration for object pushing."""

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

from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.util.terrain import (
    BackgroundTerrainImporterCfg,
    MovingPatchTerrainCfg,
)

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND

from . import terrain_cfg

BOX_SIZE = (1.0, 0.8, 1.0)
"""Box length along the push direction, width, and height [m]; same as the rigid-ground task."""

BOX_MASS = 10.0
"""Nominal box mass [kg]; a startup event resamples it per environment."""

MPM_VISUAL_COLOR = (0.32, 0.24, 0.21)
"""Sand RGB color, matching the fixed-bed locomotion task ``velocity_mpm/g1_29dof_mpm``."""

ARM_PUSH_POSTURE = {
    ".*_shoulder_pitch_joint": -0.60,
    "left_shoulder_roll_joint": 0.30,
    "right_shoulder_roll_joint": -0.30,
    "left_shoulder_yaw_joint": -0.20,
    "right_shoulder_yaw_joint": 0.20,
    ".*_elbow_joint": 0.80,
    "left_wrist_roll_joint": -0.07,
    "right_wrist_roll_joint": 0.07,
    ".*_wrist_pitch_joint": 0.27,
    "left_wrist_yaw_joint": -0.10,
    "right_wrist_yaw_joint": 0.10,
}
"""Default arm joint positions [rad]: both wrists held ahead of the chest as a bumper.

Identical to the rigid-ground task. Actions are offsets from the default joint positions,
so the fine-tuned policy needs the same defaults.
"""

# Keep the leg and waist defaults of the asset; replace only the arm joints.
ROBOT_CFG = UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND.copy()
ROBOT_CFG.init_state.joint_pos = {
    expression: value
    for expression, value in ROBOT_CFG.init_state.joint_pos.items()
    if not any(part in expression for part in ("shoulder", "elbow", "wrist"))
} | ARM_PUSH_POSTURE

SIMULATED_PATCH_SIZE = (2.8, 2.0)
"""Simulated sand patch along x and y [m].

The patch is centered between the pelvis and the box and is aligned with the world axes;
the robot pushes along x. The x size holds the trailing feet and the far face of the
1 m box at the largest reset distance and yaw, with the termination margins.
"""

VOXEL_SIZE = 0.04
"""MPM voxel size [m]."""
SAND_DEPTH = 0.25
"""Additional depth of the sand layer above the ground [m]."""
FOOT_CONTACT_MARGIN = VOXEL_SIZE * 0.5
"""Sole contact margin [m], retained with the existing policy/physics settings."""


@configclass
class ObjectPushTerrainCfg(MovingPatchTerrainCfg):
    """Recycling patch centered between the robot and one rigid object."""

    # Keep the computed property visible to configclass's per-class deepcopy pass.
    total_patch_size = MovingPatchTerrainCfg.total_patch_size

    tracked_object_body: str = "Box"
    """Exact Newton body name of the object sharing the robot's particle patch."""


@configclass
class G1ObjectPushSceneCfg(InteractiveSceneCfg):
    """G1 and a box supported by one recycling patch around the robot."""

    terrain: BackgroundTerrainImporterCfg = BackgroundTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        terrain_generator=terrain_cfg.FLAT_TERRAINS_CFG,
        max_init_terrain_level=0,
        use_terrain_origins=False,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
        moving_patch_terrain=ObjectPushTerrainCfg(
            simulated_terrain_size=SIMULATED_PATCH_SIZE,
            boundary_terrain_size=0.2,
            tracked_body="pelvis",
            robot_spawn_height=0.76,
            patch_discretization_step=0.1,
            particle_depth=SAND_DEPTH,
            voxel_size=VOXEL_SIZE,
            particles_per_cell=1.25,
            jitter=0.01,
            material=MPMParticleMaterialCfg(
                density=1600.0,
                young_modulus=1.0e15,
                friction=0.48,
                yield_pressure=1.0e15,
                tensile_yield_ratio=0.0,
                yield_stress=0.0,
                hardening=0.0,
                dilatancy=0.0,
                viscosity=0.0,
            ),
            floor_thickness=0.1,
            show_boundary_particles=True,
            visual_color=MPM_VISUAL_COLOR,
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ROBOT_CFG.spawn.replace(  # type: ignore
            # inflate the contact margin for the ankle roll links
            collision_props={
                r"/.*_ankle_roll_link/.*": [
                    NewtonCollisionCfg(
                        contact_margin=FOOT_CONTACT_MARGIN,
                        contact_gap=0.0,
                    ),
                ],
            },
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

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.1, 0.0, BOX_SIZE[2] / 2 + 0.01)),
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            rigid_props=UsdPhysicsRigidBodyCfg(),
            collision_props=[UsdPhysicsCollisionCfg(), NewtonCollisionCfg(contact_margin=0.005, contact_gap=0.0)],
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.75)),
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
