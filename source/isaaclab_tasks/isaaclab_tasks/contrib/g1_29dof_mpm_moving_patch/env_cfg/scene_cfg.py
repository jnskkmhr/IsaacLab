# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 assets, geometry and material configuration for moving terrain."""

import math

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.hf_terrains_cfg import HfWaveTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG

from ..util.terrain import BackgroundTerrainImporterCfg, MovingPatchTerrainCfg
from ..util.visual_terrain import TexturedTerrainImporterCfg

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


GENERATOR = TerrainGeneratorCfg(
    seed=42,
    size=(30.0, 30.0),
    num_rows=3,
    num_cols=1,
    border_width=0.0,
    horizontal_scale=0.1,
    curriculum=False,
    sub_terrains={
        # One tile is selected by these weights; entries are not blended.
        # To select waves, set flat.proportion=0.0 and waves.proportion=1.0.
        "flat": MeshPlaneTerrainCfg(),
        "waves": HfWaveTerrainCfg(
            amplitude_range=(0.4, 0.4),
            num_waves=4,
            border_width=0.5,
        ),
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.1, 0.4),
            num_waves=4,
            border_width=0.25,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


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
        # terrain_generator=GENERATOR,
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
        moving_patch_terrain=MovingPatchTerrainCfg(
            moving_terrain_size=(1.3, 1.3),
            boundary_terrain_size=0.2,
            tracked_body="pelvis",
            particle_depth=0.25,
            robot_spawn_height=0.76,
            shift_step=0.2,
            voxel_size=VOXEL_SIZE,
            particles_per_cell=1.25,
            jitter=0.05,
            material=MPMParticleMaterialCfg(
                density=2700.0,
                young_modulus=15.0e6,
                poisson_ratio=0.3,
                friction=math.tan(math.radians(40.0)),
                yield_pressure=1.0e12,
            ),
            floor_thickness=0.1,
            show_boundary_particles=False,
            visual_color=(0.72, 0.55, 0.34)
        ),
    )
    visual_terrain: TerrainImporterCfg = TexturedTerrainImporterCfg(
        prim_path="/World/visual_terrain",
        collision_group=-1,
        terrain_type="generator",
        disable_collider=True,
        use_terrain_origins=False,
        # terrain_generator=GENERATOR.copy(), # type: ignore
        terrain_generator=ROUGH_TERRAINS_CFG.copy(), # type: ignore
        mesh_origin_offset=(0.0, 0.0, -0.15),
    )
    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
        # init_state=UNITREE_G1_29DOF_BOX_FOOT_CFG.init_state.replace(  # type: ignore
        #     pos=(0.0, 0.0, MovingPatchTerrainCfg().robot_spawn_height),
        # ),
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
        x, y = terrain.patch_size
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
