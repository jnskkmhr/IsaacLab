# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 assets, geometry and material configuration for moving terrain."""

import math
from typing import NamedTuple

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG

from ..util.terrain import BackgroundTerrainImporterCfg, MovingPatchTerrainCfg
from ..util.visual_terrain import TexturedTerrainImporterCfg
from . import terrain_cfg

MPM_COLLIDER_MARGIN = 0.0125
"""Supporting-floor MPM contact margin [m]."""
FOOT_CONTACT_MARGIN = 0.01875
"""Sole contact margin [m], retained with the existing policy/physics settings."""
FLOOR_CONTACT_MARGIN = 0.004
"""Rigid catch-floor contact margin [m]."""
FLOOR_CONTACT_GAP = 0.002
"""Rigid catch-floor contact detection gap [m]."""
VOXEL_SIZE = 0.04
"""MPM voxel size [m]."""
SAND_DEPTH = 0.25
"""Additional depth of the sand layer above the ground [m]."""


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
}

@configclass
class G1MovingPatchSceneCfg(InteractiveSceneCfg):
    """G1, shared terrain, particles and lights for the moving-patch task."""

    # NOTE: do we need camera cfg???
    # overview_camera: CameraCfg = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/OverviewCamera",
    #     width=320,
    #     height=240,
    #     data_types=["rgb"],
    #     renderer_cfg=MultiBackendRendererCfg(),
    #     spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 30.0)),
    #     # Fixed view from (3, -4, 2.5) toward (0, 0, 0.6), relative to each environment.
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(3.0, -4.0, 2.5),
    #         rot=(0.538657606, 0.179552540, 0.260309190, 0.780927658),
    #         convention="opengl",
    #     ),
    # )

    terrain: BackgroundTerrainImporterCfg = BackgroundTerrainImporterCfg(
        disable_visual=True,
        visual_material=None,
        mpm_contact_margin=MPM_COLLIDER_MARGIN,
        rigid_contact_margin=FLOOR_CONTACT_MARGIN,
        rigid_contact_gap=FLOOR_CONTACT_GAP,
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        terrain_generator=terrain_cfg.ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
        moving_patch_terrain=MovingPatchTerrainCfg(
            simulated_terrain_size=(1.2, 1.2),
            boundary_terrain_size=0.2,
            tracked_body="pelvis",
            robot_spawn_height=0.76,
            patch_discretization_step=0.2,
            particle_depth=SAND_DEPTH,
            voxel_size=VOXEL_SIZE,
            particles_per_cell=1.25,
            jitter=0.05,
            material=MPMParticleMaterialCfg(
                **{name: value for name, value in MATERIAL_PRESETS["sand"]._asdict().items()}
                # **{name: value for name, value in MATERIAL_PRESETS["snow"]._asdict().items()}
                # **{name: value for name, value in MATERIAL_PRESETS["clay"]._asdict().items()}
            ),
            floor_thickness=0.1,
            show_boundary_particles=False,
            visual_color=(0.72, 0.55, 0.34),
        ),
    )
    visual_terrain: TerrainImporterCfg = TexturedTerrainImporterCfg(
        prim_path="/World/visual_terrain",
        # disable_visual=True,
        collision_group=-1,
        terrain_type="generator",
        disable_collider=True,
        use_terrain_origins=False,
        terrain_generator=terrain_cfg.ROUGH_TERRAINS_CFG,
        mesh_origin_offset=(0.0, 0.0, -SAND_DEPTH),
    )
    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UNITREE_G1_29DOF_BOX_FOOT_CFG.spawn.replace(  # type: ignore
            # Scoped to the sole colliders. Newton sums both shapes' margins, so applying this to
            # the whole robot pushes every non-adjacent link pair apart by twice the margin; with
            # self-collisions enabled the shin and the sole sit 0.02 m apart in the nominal stance
            # and the legs lock up. Only the proxied ankle roll links are seen by the MPM solver,
            # so only they need the inflation.
            collision_props={
                r"/.*_ankle_roll_link/.*": [
                    NewtonCollisionCfg(
                        contact_margin=FOOT_CONTACT_MARGIN,
                        # Implicit MPM consumes the shape margin; gap is a rigid-contact parameter.
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
        debug_vis=True,
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
        # Generate identical visual and support geometry from one configured source.
        if self.terrain.terrain_generator.seed is None:
            raise ValueError("Set terrain_generator.seed so visual and support terrains match")

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
