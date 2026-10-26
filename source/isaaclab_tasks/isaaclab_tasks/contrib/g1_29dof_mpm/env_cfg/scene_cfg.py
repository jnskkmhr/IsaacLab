# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed sand bed with a rigid approach platform and separate support colliders.

All dimensions and positions below are in the environment frame, in meters.
The initial sand surface is at z = 0. The MPM floor/walls retain particles;
MJWarp owns the approach platform and a floor beneath the sand bed.
"""

import math

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg, NewtonCollisionPropertiesCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG

from ..util.terrain import MeshXDuneTerrainCfg, MeshXSlopeTerrainCfg, TerrainMPMGridCfg
from ..util.tilting_testbed import TiltingTestbedSpawnerCfg

# Particle/material settings match the moving-patch sand configuration.
MPM_VOXEL_SIZE = 0.04
MPM_COLLIDER_MARGIN = 0.0
# FOOT_CONTACT_MARGIN = 0.01875
FOOT_CONTACT_MARGIN = MPM_VOXEL_SIZE * 0.5
# FOOT_CONTACT_MARGIN = MPM_VOXEL_SIZE * 0.3
PLATFORM_CONTACT_MARGIN = 0.004
PLATFORM_CONTACT_GAP = 0.002
MPM_PARTICLES_PER_CELL = 1.25
MPM_VISUAL_COLOR = (0.32, 0.24, 0.21)

# SAND_MATERIAL_CFG = MPMParticleMaterialCfg(
#     density=2700.0,
#     young_modulus=1.0e15,
#     poisson_ratio=0.3,
#     friction=0.58,
#     yield_pressure=1.0e15,
#     tensile_yield_ratio=0.0,
#     yield_stress=0.0,
#     hardening=0.0,
#     dilatancy=0.0,
#     viscosity=0.0,
# )

# # Denser dry beach sand: initial tuning preset, not calibrated to a measured beach.
# # Density reference: FHWA NHI-05-037, Table 5-9 (dense dry sand).
# # https://www.fhwa.dot.gov/engineering/geotech/pubs/05037/05a.cfm
# SAND_MATERIAL_CFG = MPMParticleMaterialCfg(
#     density=1800.0,         # Bulk bed density, kg/m³, including pore space
#     young_modulus=1.0e15,   # Retain the existing stiffness approximation
#     poisson_ratio=0.3,
#     friction=0.80,          # Higher shear resistance; uncalibrated tuning value
#     yield_pressure=1.0e15,
#     tensile_yield_ratio=0.0,
#     yield_stress=0.0,
#     hardening=0.0,
#     dilatancy=0.0,
#     viscosity=0.0,
# )


# # basalt
# # K. Manick et al.,
# # “The European Space Agency Sample Analogue Collection (ESA2C) for Exploration Missions: Geotechnical Characterization” (2017).
# SAND_MATERIAL_CFG = MPMParticleMaterialCfg(
#     density=1660.0,          # Measured loose bulk density of reference aggregate
#     young_modulus=1.0e15,
#     poisson_ratio=0.3,
#     friction=0.88,           # Initial estimate: tan(41.3 degrees)
#     yield_pressure=1.0e15,
#     tensile_yield_ratio=0.0,
#     yield_stress=0.0,
#     hardening=0.0,
#     dilatancy=0.0,
#     viscosity=0.0,
# )

# # Loosely packed poppy seed
# # Yang et al. (2020)
# # “Improved Terzaghi-theory-based interaction modeling of rotary robotic locomotors with granular substrates.”
# SAND_MATERIAL_CFG = MPMParticleMaterialCfg(
#     density=640.0,           # Bulk seed-bed density, kg/m³
#     young_modulus=1.0e15,    # Retain effectively rigid grains initially
#     poisson_ratio=0.3,
#     friction=0.73,          # Initial estimate; calibrate against pile slope
#     yield_pressure=1.0e15,
#     tensile_yield_ratio=0.0,
#     yield_stress=0.0,
#     hardening=0.0,
#     dilatancy=0.0,
#     viscosity=0.0,
# )

# Closely packed poppy seeds
# Yang et al. (2020), Table 2
SAND_MATERIAL_CFG = MPMParticleMaterialCfg(
    density=682.0,          # Bulk seed-bed density, kg/m³
    young_modulus=1.0e15,
    poisson_ratio=0.3,
    friction=1.07,          # Estimate from tan(47°); requires calibration
    yield_pressure=1.0e15,
    tensile_yield_ratio=0.0,
    yield_stress=0.0,
    hardening=0.0,
    dilatancy=0.0,
    viscosity=0.0,
)


def _mpm_collider_box(
    prim_path: str, *, size: tuple[float, float, float], position: tuple[float, float, float], visible: bool = False
) -> RigidObjectCfg:
    """Create a stationary retaining box owned by the MPM solver."""
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True, contact_margin=MPM_COLLIDER_MARGIN, contact_gap=0.0
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25), roughness=0.8),
            visible=visible,
        ),
    )


def _rigid_collider_box(
    prim_path: str, *, size: tuple[float, float, float], position: tuple[float, float, float], visible: bool = False
) -> AssetBaseCfg:
    """Create a static support box owned by MJWarp."""
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True, contact_margin=PLATFORM_CONTACT_MARGIN, contact_gap=PLATFORM_CONTACT_GAP
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.37), roughness=0.8),
            visible=visible,
        ),
    )


def _visual_box(
    prim_path: str, *, size: tuple[float, float, float], position: tuple[float, float, float]
) -> AssetBaseCfg:
    """Draw the approach surface at the initial sand-surface height without adding collision."""
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.37), roughness=0.8),
            visible=True,
        ),
    )


def _sand_cfg(
    size: tuple[float, float, float],
    *,
    jitter: float = 0.05,
    terrain: terrain_gen.SubTerrainBaseCfg | None = None,
) -> MPMObjectCfg:
    """Create a scene's particle grid, optionally following a single generated terrain tile."""
    if len(size) != 3 or any(not math.isfinite(extent) or extent <= 0.0 for extent in size):
        raise ValueError("Sand-bed dimensions must be positive and finite [m].")
    particle_params = dict(
        lower=(-size[0] / 2, -size[1] / 2, -size[2]),
        upper=(size[0] / 2, size[1] / 2, 0.0),
        voxel_size=MPM_VOXEL_SIZE,
        particles_per_cell=MPM_PARTICLES_PER_CELL,
        particle_placement="cell_center",
        jitter=jitter,
        material=SAND_MATERIAL_CFG,
        visual_color=MPM_VISUAL_COLOR,
    )
    if terrain is None:
        spawn = MPMGridCfg(**particle_params)
    else:
        spawn = TerrainMPMGridCfg(
            **particle_params,
            terrain_generator=terrain_gen.TerrainGeneratorCfg(
                seed=42,
                size=size[:2],
                num_rows=1,
                num_cols=1,
                border_width=0.0,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                use_cache=False,
                sub_terrains={"slope" if isinstance(terrain, MeshXSlopeTerrainCfg) else "dune": terrain},
            ),
        )
    return MPMObjectCfg(prim_path="{ENV_REGEX_NS}/Sand", spawn=spawn)


@configclass
class G1MPMSceneCfg(InteractiveSceneCfg):
    """G1 walking from a rigid approach onto a 6 x 2 m flat sand bed.

    Each scene declares its own sand size. Particle bounds are the source of truth
    for support geometry, approach placement and robot placement.
    """

    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UNITREE_G1_29DOF_BOX_FOOT_CFG.spawn.replace(
            collision_props={
                "/.*_ankle_roll_link/.*": [NewtonCollisionCfg(contact_margin=FOOT_CONTACT_MARGIN, contact_gap=0.0)]
            }
        ),
    )
    sand = _sand_cfg(size=(4.0, 2.0, 0.3))
    # The approach length includes the 0.1 m entry retaining wall; its width is fitted to the bed.
    approach_platform = _rigid_collider_box(
        "{ENV_REGEX_NS}/ApproachPlatform", size=(3.1, 2.2, 0.5), position=(0.0, 0.0, 0.0)
    )
    approach_platform_visual: AssetBaseCfg | None = None
    bed_floor: RigidObjectCfg | None = None
    bed_wall_front: RigidObjectCfg | None = None
    bed_wall_left: RigidObjectCfg | None = None
    bed_wall_right: RigidObjectCfg | None = None
    approach_bank: RigidObjectCfg | None = None
    rigid_bed_floor: AssetBaseCfg | None = None
    testbed: ArticulationCfg | None = None
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self) -> None:
        lower, upper = self.sand.spawn.lower, self.sand.spawn.upper
        length, width, depth = (hi - lo for lo, hi in zip(lower, upper, strict=True))
        if self.testbed is not None:
            self.testbed.spawn.sand_bed_size = (length, width, depth)
            self.testbed.init_state.pos = (0.0, 0.0, lower[2])
            self.robot.init_state.pos = (0.0, 0.0, upper[2] + 0.76)
            return

        wall, wall_height, floor_thickness = 0.1, 0.2, 0.1
        floor_size = (length + 2 * wall, width + 2 * wall, floor_thickness)
        floor_position = (0.0, 0.0, lower[2] - floor_thickness / 2)
        # The approach top is 5 cm below the initial sand surface, matching the settled default sand.
        approach_top = upper[2] - 0.05
        approach = self.approach_platform
        approach_length, _, approach_thickness = approach.spawn.size
        approach.spawn.size = (approach_length, width + 2 * wall, approach_thickness)
        approach.init_state.pos = (lower[0] - approach_length / 2, 0.0, approach_top - approach_thickness / 2)
        self.approach_platform_visual = _visual_box(
            "{ENV_REGEX_NS}/ApproachPlatformVisual", size=approach.spawn.size, position=approach.init_state.pos
        )
        self.robot.init_state.pos = (lower[0] - (approach_length - wall) / 2, 0.0, approach_top + 0.76)
        if isinstance(self.sand.spawn, TerrainMPMGridCfg):
            self.sand.spawn.terrain_generator.size = (length, width)
            self.sand.spawn.entrance_height = approach_top
            return

        self.bed_floor = _mpm_collider_box(
            "{ENV_REGEX_NS}/MPMBedFloor", size=floor_size, position=floor_position, visible=True
        )
        self.rigid_bed_floor = _rigid_collider_box(
            "{ENV_REGEX_NS}/RigidBedFloor", size=floor_size, position=floor_position
        )
        self.bed_wall_front = _mpm_collider_box(
            "{ENV_REGEX_NS}/MPMBedWallFront",
            size=(wall, width + 2 * wall, wall_height),
            position=(upper[0] + wall / 2, 0.0, lower[2] + wall_height / 2),
            visible=True,
        )
        for name, side in (("left", 1), ("right", -1)):
            setattr(
                self,
                f"bed_wall_{name}",
                _mpm_collider_box(
                    f"{{ENV_REGEX_NS}}/MPMBedWall{name.title()}",
                    size=(length + 2 * wall, wall, wall_height),
                    position=(0.0, side * (width + wall) / 2, lower[2] + wall_height / 2),
                    visible=True,
                ),
            )
        bank_height = approach_top - lower[2] + floor_thickness
        self.approach_bank = _mpm_collider_box(
            "{ENV_REGEX_NS}/MPMApproachBank",
            size=(approach_length, width + 2 * wall, bank_height),
            position=(approach.init_state.pos[0], 0.0, approach_top - bank_height / 2),
        )


@configclass
class G1RoughMPMSceneCfg(G1MPMSceneCfg):
    """A 6 x 2 m ascending ramp, with a terrain-conforming sand layer."""

    sand = _sand_cfg(
        size=(6.0, 2.0, 0.3),
        jitter=0.016,
        terrain=MeshXSlopeTerrainCfg(proportion=1.0, slope_range=(0.4, 0.4)),
    )


@configclass
class G1DescendingMPMSceneCfg(G1RoughMPMSceneCfg):
    """A 6 x 2 m ramp descending along +X from the approach platform."""

    sand = _sand_cfg(
        size=(6.0, 2.0, 0.3),
        jitter=0.016,
        terrain=MeshXSlopeTerrainCfg(proportion=1.0, slope_range=(-0.4, -0.4)),
    )


@configclass
class G1DuneMPMSceneCfg(G1RoughMPMSceneCfg):
    """Two smooth dune ridges along X on a 6 x 2 m bed."""

    sand = _sand_cfg(
        size=(6.0, 2.0, 0.3),
        jitter=0.016,
        terrain=MeshXDuneTerrainCfg(proportion=1.0, height_range=(0.3, 0.3), num_waves=2),
    )


@configclass
class G1TiltingMPMSceneCfg(G1MPMSceneCfg):
    """G1 starts in a 3 x 2 m powered sand tray for balance evaluation."""

    sand = _sand_cfg(size=(3.0, 2.0, 0.3))
    approach_platform = None
    testbed: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Testbed",
        spawn=TiltingTestbedSpawnerCfg(wall_thickness=0.1, floor_thickness=0.1),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}),
        actuators={
            "tilt": ImplicitActuatorCfg(
                joint_names_expr=["roll", "pitch"],
                stiffness=1.0e7,
                damping=2.0e5,
                effort_limit_sim=1.0e6,
                velocity_limit_sim=0.5,
                armature=1000.0,
            ),
        },
    )


@configclass
class G1WalkingTiltingMPMSceneCfg(G1TiltingMPMSceneCfg):
    """A square 4 x 4 m powered sand tray for forward/lateral walking and turning."""

    sand = _sand_cfg(size=(4.0, 4.0, 0.3))
