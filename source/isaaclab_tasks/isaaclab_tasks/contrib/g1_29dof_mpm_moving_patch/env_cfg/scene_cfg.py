# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 assets, geometry and material configuration for moving terrain."""

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfWaveTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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


GENERATOR = TerrainGeneratorCfg(
    seed=42,
    size=(30.0, 30.0),
    num_rows=1,
    num_cols=1,
    border_width=0.0,
    horizontal_scale=0.1,
    curriculum=False,
    sub_terrains={
        # One tile is selected by these weights; entries are not blended.
        # To select waves, set flat.proportion=0.0 and waves.proportion=1.0.
        # "flat": MeshPlaneTerrainCfg(proportion=1.0),
        "waves": HfWaveTerrainCfg(
            # proportion=0.0,
            amplitude_range=(0.4, 0.4),
            num_waves=4,
            # Keep height-field edge correction inside a flat perimeter [m].
            border_width=0.5,
        ),
    },
)


@configclass
class G1MovingPatchSceneCfg(InteractiveSceneCfg):
    """G1, shared terrain, particles and lights for the moving-patch task."""

    terrain: BackgroundTerrainImporterCfg = BackgroundTerrainImporterCfg(
        disable_visual=True,
        visual_material=None,
        mpm_contact_margin=MPM_COLLIDER_MARGIN,
        rigid_contact_margin=FLOOR_CONTACT_MARGIN,
        rigid_contact_gap=FLOOR_CONTACT_GAP,
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        terrain_generator=GENERATOR,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
    )
    visual_terrain: TerrainImporterCfg = TexturedTerrainImporterCfg(
        prim_path="/World/visual_terrain",
        collision_group=-1,
        terrain_type="generator",
        disable_collider=True,
        use_terrain_origins=False,
        terrain_generator=GENERATOR.copy(), # type: ignore
    )
    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_G1_29DOF_BOX_FOOT_CFG.init_state.replace(  # type: ignore
            pos=(0.0, 0.0, MovingPatchTerrainCfg().robot_spawn_height),
        ),
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
