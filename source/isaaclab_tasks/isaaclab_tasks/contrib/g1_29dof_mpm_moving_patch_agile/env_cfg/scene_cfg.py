# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 assets, geometry and material configuration for moving terrain."""

from typing import NamedTuple

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim.spawners.materials import PhysxRigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.terrain import (
    BackgroundTerrainImporterCfg,
    MovingPatchTerrainCfg,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.scene_cfg import G1_29DOF_BeyondMimic_delay

from . import terrain_cfg

VOXEL_SIZE = 0.04
"""MPM voxel size [m]."""
SAND_DEPTH = 0.25
"""Additional depth of the sand layer above the ground [m]."""
FOOT_CONTACT_MARGIN = VOXEL_SIZE * 0.5
"""Foot collider margin in MPM environments [m]."""


class StripMaterialPreset(NamedTuple):
    """Material values for one particle strip."""

    density: float
    young_modulus: float
    friction: float
    yield_pressure: float
    tensile_yield_ratio: float
    yield_stress: float
    hardening: float
    dilatancy: float
    viscosity: float


MATERIAL_PRESETS = {
    # Values start from the sand, snow, and mud rows of Table 5 in Daviet's
    # mixed-MPM paper. A finite 1 PPa value represents the tabulated rigid
    # elastic limit while remaining valid input to Newton's schema.
    "sand": StripMaterialPreset(1600.0, 1.0e15, 0.48, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0),
    "snow": StripMaterialPreset(250.0, 1.0e15, 0.30, 2.0e6, 0.05, 0.0, 1.0, 1.0, 0.0),
    "clay": StripMaterialPreset(1500.0, 1.0e15, 0.0, 1.0e15, 1.0, 200.0, 0.0, 0.1, 100.0),
    # Synthetic stiff, cohesive material for rigid-ground policy comparisons, not a sand calibration.
    # Finite elasticity and high compressive, tensile, and shear yield limits resist rearrangement.
    "rigid": StripMaterialPreset(
        density=1600.0,
        young_modulus=1.0e8,
        friction=0.48,
        yield_pressure=1.0e9,
        tensile_yield_ratio=1.0,
        yield_stress=1.0e9,
        hardening=0.0,
        dilatancy=0.0,
        viscosity=0.0,
    ),
}

MATERIAL_NAME = "sand"


@configclass
class G1MovingPatchSceneCfg(InteractiveSceneCfg):
    """G1, shared terrain, particles and lights for the moving-patch task."""

    terrain: BackgroundTerrainImporterCfg = BackgroundTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        terrain_generator=terrain_cfg.ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
        physics_material=PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        moving_patch_terrain=MovingPatchTerrainCfg(
            simulated_terrain_size=(1.2, 1.2),
            boundary_terrain_size=0.2,
            tracked_body="pelvis",
            robot_spawn_height=G1_29DOF_BeyondMimic_delay.init_state.pos[2],
            patch_discretization_step=0.2,
            particle_depth=SAND_DEPTH,
            voxel_size=VOXEL_SIZE,
            particles_per_cell=1.25,
            jitter=0.05,
            material=MPMParticleMaterialCfg(**MATERIAL_PRESETS[MATERIAL_NAME]._asdict()),
            floor_thickness=0.1,
            show_boundary_particles=False,
            visual_color=(0.72, 0.55, 0.34),
        ),
    )
    robot: ArticulationCfg = G1_29DOF_BeyondMimic_delay.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        # The moving-patch MODEL_INIT callback assigns per-terrain foot margins.
        newton_collision_shape_margin=None,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2)),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=False,
    )

    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
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
