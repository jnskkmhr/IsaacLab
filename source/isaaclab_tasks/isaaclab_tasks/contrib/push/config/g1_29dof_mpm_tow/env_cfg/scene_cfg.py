# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1, a sled on a rope, flat terrain and the sand layer of the tow task."""

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import StreamingCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, StreamingMPMGridCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.velocity_mpm.util.terrain import MPMTerrainImporterCfg, SandLayerCfg

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG

from . import terrain_generator_cfg

VOXEL_SIZE = 0.04
"""MPM voxel size [m]."""
SAND_DEPTH = 0.25
"""Additional depth of the sand layer above the ground [m]."""

SAND_MATERIAL = MPMParticleMaterialCfg(
    # Loosely packed poppy seed, the sand of the walking task's playback configuration. One fixed
    # material: this task does not randomize the sand.
    density=640.0,
    young_modulus=1.0e15,
    friction=0.73,
    yield_pressure=1.0e15,
    tensile_yield_ratio=0.0,
    yield_stress=0.0,
    hardening=0.0,
    dilatancy=0.0,
    viscosity=0.0,
)
# SAND_MATERIAL = MPMParticleMaterialCfg(
#     # Loosely packed poppy seed, the sand of the walking task's playback configuration. One fixed
#     # material: this task does not randomize the sand.
#     density=682.0,
#     young_modulus=1.0e15,
#     friction=1.07,
#     yield_pressure=1.0e15,
#     tensile_yield_ratio=0.0,
#     yield_stress=0.0,
#     hardening=0.0,
#     dilatancy=0.0,
#     viscosity=0.0,
# )
"""Sand material of every environment."""

SLED_ASSET = "sled"
"""Scene name of the towed object."""
SLED_NAME = "Sled"
"""Prim and body name of the towed object; the streaming pool and the activity region find it by this name."""
SLED_SIZE = (0.3, 0.3, 0.3)
"""Length along the tow direction, width and height of the sled [m]."""
SLED_MASS = 15.0
"""Mass of the sled [kg]."""
SLED_FRICTION = 0.5
"""Friction coefficient of the sled surface [-]."""
SLED_RESET_DISTANCE = 1.15
"""Distance of the sled center behind the pelvis at reset [m]; the rope is just slack there."""


@configclass
class G1TowSceneCfg(InteractiveSceneCfg):
    """G1, shared flat terrain, sand layer, and a sled resting on the sand behind the robot."""

    terrain: MPMTerrainImporterCfg = MPMTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        terrain_generator=terrain_generator_cfg.FLAT_TERRAINS_CFG,
        max_init_terrain_level=0,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
        sand=SandLayerCfg(
            robot_spawn_height=0.76,
            particle_depth=SAND_DEPTH,
            voxel_size=VOXEL_SIZE,
            particles_per_cell=1.25,
            jitter=0.05,
            material=SAND_MATERIAL,
            floor_thickness=0.1,
            show_boundary_particles=False,
            visual_color=(0.72, 0.55, 0.34),
        ),
    )
    # The robot keeps the collision properties of its asset. The distance the sand keeps from the feet is set in
    # the MPM solver configuration (MPMSolverCfg.collider_margins), which the rigid contact does not read.
    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
    sled: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + SLED_NAME,
        spawn=sim_utils.CuboidCfg(
            size=SLED_SIZE,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=SLED_MASS),
            collision_props=[
                UsdPhysicsCollisionCfg(),
                # No margin on the sled's rigid contact; the sand's margin is MPMSolverCfg.collider_margins.
                NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.0),
            ],
            physics_material=RigidBodyMaterialBaseCfg(static_friction=SLED_FRICTION, dynamic_friction=SLED_FRICTION),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.75), roughness=0.6),
        ),
        # Behind the robot, just above the sand surface; the reset event places it relative to the robot.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-SLED_RESET_DISTANCE, 0.0, 0.5 * SLED_SIZE[2] + 0.02)),
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

    def configure_sand(self, streaming: StreamingCfg | None) -> None:
        """Derive the robot spawn height and the particle pool from the final sand and streaming settings."""
        if streaming is None:
            raise ValueError(
                "The terrain scene covers the ground with a streaming pool; set MPMSolverCfg.streaming, "
                "or use a testbed scene to simulate every particle of a fixed bed."
            )
        self.terrain.validate_geometry()

        sand = self.terrain.sand
        x, y = streaming.pool_size
        # TerrainImporter supplies each robot's world-space spawn origin and height.
        self.robot.init_state.pos = (0.0, 0.0, sand.robot_spawn_height)
        grid = dict(
            lower=(-x / 2, -y / 2, -sand.particle_depth),
            upper=(x / 2, y / 2, 0.0),
            voxel_size=sand.voxel_size,
            particles_per_cell=sand.particles_per_cell,
            particle_placement="cell_center",
            jitter=sand.jitter,
            material=sand.material,
            visual_color=sand.visual_color,
        )
        if streaming.cache is None:
            spawn = MPMGridCfg(**grid)
        else:
            # A tile cache needs spare simulation slots for sand that accumulates in a tile.
            spawn = StreamingMPMGridCfg(
                **grid,
                tile_size=streaming.tile_size,
                simulation_capacity_multiplier=streaming.cache.simulation_capacity_multiplier,
                # The visualizers submit the sand surface themselves. An authored Points visual
                # would also be rendered by RTX's borrowed stage.
                visible=sand.render_mode != "isosurface",
            )
        self.sand = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            init_state=MPMObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            spawn=spawn,
        )
