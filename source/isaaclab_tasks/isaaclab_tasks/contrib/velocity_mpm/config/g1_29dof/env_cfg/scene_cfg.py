# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed sand bed with separate support colliders; the robot starts on the sand.

All dimensions and positions below are in the environment frame, in meters.
The initial sand surface is at z = 0. The MPM floor/walls retain particles;
MJWarp owns a floor beneath the sand bed.
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

from ....util.testbed import (
    MeshXDuneTerrainCfg,
    MeshXSlopeTerrainCfg,
    TerrainMPMGridCfg,
    TiltingTestbedSpawnerCfg,
    terrain_height_from_entrance,
)
from .streaming_scene_cfg import StripMaterialPreset

# Particle/material settings match the moving-patch sand configuration.
MPM_VOXEL_SIZE = 0.04
# MPM_VOXEL_SIZE = 0.02
MPM_COLLIDER_MARGIN = 0.0
FOOT_CONTACT_MARGIN = 0.04 * 0.5
PLATFORM_CONTACT_MARGIN = 0.004
PLATFORM_CONTACT_GAP = 0.002
MPM_PARTICLES_PER_CELL = 1.25
MPM_VISUAL_COLOR = (0.32, 0.24, 0.21)
ROBOT_SPAWN_INSET = 1.0
"""Distance of the robot's start position from the negative-X end of the bed [m]."""

MATERIAL_PRESETS = {
    # Each entry lists density [kg/m^3], Young's modulus [Pa], friction, yield pressure [Pa], tensile yield
    # ratio, yield stress [Pa], hardening, dilatancy and viscosity; the Poisson ratio is the default 0.3.
    # Unlabelled alternative kept from the earlier fixed-bed definitions.
    "sand_2700": StripMaterialPreset(2700.0, 1.0e15, 0.58, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0),
    # Denser dry beach sand: initial tuning preset, not calibrated to a measured beach. The density is the
    # bulk bed density including pore space; the friction is an uncalibrated tuning value.
    # Density reference: FHWA NHI-05-037, Table 5-9 (dense dry sand).
    # https://www.fhwa.dot.gov/engineering/geotech/pubs/05037/05a.cfm
    "beach_sand": StripMaterialPreset(1800.0, 1.0e15, 0.80, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0),
    # Basalt: measured loose bulk density of the reference aggregate; friction is an initial estimate,
    # tan(41.3 degrees). K. Manick et al., "The European Space Agency Sample Analogue Collection (ESA2C)
    # for Exploration Missions: Geotechnical Characterization" (2017).
    "basalt": StripMaterialPreset(1660.0, 1.0e15, 0.88, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0),
    # Loosely packed poppy seed: bulk seed-bed density; friction is an initial estimate to calibrate
    # against the pile slope. Yang et al. (2020), "Improved Terzaghi-theory-based interaction modeling of
    # rotary robotic locomotors with granular substrates."
    "poppy_seed_loose": StripMaterialPreset(640.0, 1.0e15, 0.73, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0),
    # Closely packed poppy seeds: bulk seed-bed density; friction estimated from tan(47 degrees), requires
    # calibration. Yang et al. (2020), Table 2.
    "poppy_seed_packed": StripMaterialPreset(682.0, 1.0e15, 1.07, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0),
}

MATERIAL_NAME = "poppy_seed_packed"

SAND_MATERIAL_CFG = MPMParticleMaterialCfg(**MATERIAL_PRESETS[MATERIAL_NAME]._asdict())


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
class G1TestbedSceneCfg(InteractiveSceneCfg):
    """G1 walking on a 4 x 2 m flat sand bed, starting near its negative-X end.

    Each scene declares its own sand size. Particle bounds are the source of truth
    for support geometry and robot placement.
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
    bed_floor: RigidObjectCfg | None = None
    bed_wall_front: RigidObjectCfg | None = None
    bed_wall_back: RigidObjectCfg | None = None
    bed_wall_left: RigidObjectCfg | None = None
    bed_wall_right: RigidObjectCfg | None = None
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
        # The robot starts on the sand, inset from the negative-X end so the bed lies ahead of it.
        self.robot.init_state.pos = (lower[0] + ROBOT_SPAWN_INSET, 0.0, upper[2] + 0.76)
        if isinstance(self.sand.spawn, TerrainMPMGridCfg):
            spawn = self.sand.spawn
            spawn.terrain_generator.size = (length, width)
            # The surface is at zero height along the negative-X edge; that wall rises like the others.
            spawn.entrance_height = spawn.wall_height
            (terrain,) = spawn.terrain_generator.sub_terrains.values()
            surface = terrain_height_from_entrance(terrain, (length, width), ROBOT_SPAWN_INSET)
            self.robot.init_state.pos = (lower[0] + ROBOT_SPAWN_INSET, 0.0, upper[2] + surface + 0.76)
            return

        self.bed_floor = _mpm_collider_box(
            "{ENV_REGEX_NS}/MPMBedFloor", size=floor_size, position=floor_position, visible=True
        )
        self.rigid_bed_floor = _rigid_collider_box(
            "{ENV_REGEX_NS}/RigidBedFloor", size=floor_size, position=floor_position
        )
        for name, x in (("front", upper[0] + wall / 2), ("back", lower[0] - wall / 2)):
            setattr(
                self,
                f"bed_wall_{name}",
                _mpm_collider_box(
                    f"{{ENV_REGEX_NS}}/MPMBedWall{name.title()}",
                    size=(wall, width + 2 * wall, wall_height),
                    position=(x, 0.0, lower[2] + wall_height / 2),
                    visible=True,
                ),
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


@configclass
class G1UphillTestbedSceneCfg(G1TestbedSceneCfg):
    """A 6 x 2 m uphill ramp, with a terrain-conforming sand layer."""

    sand = _sand_cfg(
        size=(6.0, 2.0, 0.3),
        jitter=0.016,
        terrain=MeshXSlopeTerrainCfg(proportion=1.0, slope_range=(math.radians(10.0), math.radians(10.0))),
    )


@configclass
class G1DownhillTestbedSceneCfg(G1UphillTestbedSceneCfg):
    """A 6 x 2 m ramp downhill along +X."""

    sand = _sand_cfg(
        size=(6.0, 2.0, 0.3),
        jitter=0.016,
        terrain=MeshXSlopeTerrainCfg(proportion=1.0, slope_range=(-math.radians(10.0), -math.radians(10.0))),
    )


@configclass
class G1DuneTestbedSceneCfg(G1UphillTestbedSceneCfg):
    """Two smooth dune ridges along X on a 6 x 2 m bed."""

    sand = _sand_cfg(
        size=(6.0, 2.0, 0.3),
        jitter=0.016,
        terrain=MeshXDuneTerrainCfg(proportion=1.0, height_range=(0.3, 0.3), num_waves=2),
    )


@configclass
class G1TiltingTestbedSceneCfg(G1TestbedSceneCfg):
    """G1 starts in a 3 x 2 m powered sand tray for balance evaluation."""

    sand = _sand_cfg(size=(3.0, 2.0, 0.3))
    testbed: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Testbed",
        spawn=TiltingTestbedSpawnerCfg(wall_thickness=0.1, floor_thickness=0.1),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}),
        actuators={
            "tilt": ImplicitActuatorCfg(
                joint_names_expr=["roll", "pitch"],
                stiffness=1.0e7,
                damping=2.0e5,
                joint_effort_limit=1.0e6,
                joint_velocity_limit=0.5,
                armature=1000.0,
            ),
        },
    )


@configclass
class G1WalkingTiltingTestbedSceneCfg(G1TiltingTestbedSceneCfg):
    """A square 4 x 4 m powered sand tray for forward/lateral walking and turning."""

    sand = _sand_cfg(size=(4.0, 4.0, 0.3))
