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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG

# Particle/material settings match the moving-patch sand configuration.
MPM_VOXEL_SIZE = 0.04
MPM_COLLIDER_MARGIN = 0.0
FOOT_CONTACT_MARGIN = 0.01875
MPM_PARTICLES_PER_CELL = 1.25
MPM_PARTICLE_SPACING = MPM_VOXEL_SIZE / MPM_PARTICLES_PER_CELL
MPM_VISUAL_COLOR = (0.32, 0.24, 0.21)

# Sand bed and retaining-pan geometry.
SAND_BED_SIZE = (3.0, 3.0, 0.25)
SAND_SURFACE_Z = 0.0
BED_WALL_THICKNESS = 0.1
BED_WALL_HEIGHT = 0.4
BED_FLOOR_THICKNESS = 0.1
_BED_FLOOR_TOP_Z = SAND_SURFACE_Z - SAND_BED_SIZE[2]
_BED_FLOOR_CENTER_Z = _BED_FLOOR_TOP_Z - 0.5 * BED_FLOOR_THICKNESS
_BED_WALL_CENTER_Z = _BED_FLOOR_TOP_Z + 0.5 * BED_WALL_HEIGHT
BED_FLOOR_SIZE = (
    SAND_BED_SIZE[0] + 2.0 * BED_WALL_THICKNESS,
    SAND_BED_SIZE[1] + 2.0 * BED_WALL_THICKNESS,
    BED_FLOOR_THICKNESS,
)

BED_FLOOR_POSITION = (0.0, 0.0, _BED_FLOOR_CENTER_Z)
BED_WALL_X_SIZE = (BED_WALL_THICKNESS, BED_FLOOR_SIZE[1], BED_WALL_HEIGHT)
BED_WALL_Y_SIZE = (BED_FLOOR_SIZE[0], BED_WALL_THICKNESS, BED_WALL_HEIGHT)
_BED_WALL_X_OFFSET = 0.5 * (SAND_BED_SIZE[0] + BED_WALL_THICKNESS)
_BED_WALL_Y_OFFSET = 0.5 * (SAND_BED_SIZE[1] + BED_WALL_THICKNESS)
SAND_BED_XY_BOUNDS = (
    (-0.5 * SAND_BED_SIZE[0], 0.5 * SAND_BED_SIZE[0]),
    (-0.5 * SAND_BED_SIZE[1], 0.5 * SAND_BED_SIZE[1]),
)

# The hidden approach collider is recessed by the combined foot/platform margin.
# Its visual surface stays level with the initial sand surface.
APPROACH_LENGTH = 3.0
APPROACH_THICKNESS = 0.5
PLATFORM_CONTACT_MARGIN = 0.004
PLATFORM_CONTACT_GAP = 0.002
APPROACH_SURFACE_Z = SAND_SURFACE_Z - (FOOT_CONTACT_MARGIN + PLATFORM_CONTACT_MARGIN)
_APPROACH_X_HI = -0.5 * SAND_BED_SIZE[0]
_APPROACH_X_LO = _APPROACH_X_HI - BED_WALL_THICKNESS - APPROACH_LENGTH
_APPROACH_CENTER_X = 0.5 * (_APPROACH_X_LO + _APPROACH_X_HI)
_APPROACH_SIZE_X = _APPROACH_X_HI - _APPROACH_X_LO
APPROACH_BANK_HEIGHT = SAND_BED_SIZE[2] + BED_FLOOR_THICKNESS
APPROACH_BANK_SIZE = (_APPROACH_SIZE_X, BED_FLOOR_SIZE[1], APPROACH_BANK_HEIGHT)
APPROACH_BANK_POSITION = (_APPROACH_CENTER_X, 0.0, SAND_SURFACE_Z - 0.5 * APPROACH_BANK_HEIGHT)
APPROACH_PLATFORM_SIZE = (_APPROACH_SIZE_X, BED_FLOOR_SIZE[1], APPROACH_THICKNESS)
APPROACH_PLATFORM_POSITION = (_APPROACH_CENTER_X, 0.0, APPROACH_SURFACE_Z - 0.5 * APPROACH_THICKNESS)
APPROACH_VISUAL_POSITION = (_APPROACH_CENTER_X, 0.0, SAND_SURFACE_Z - 0.5 * APPROACH_THICKNESS)
ROBOT_SPAWN_X = _APPROACH_X_HI - 0.5 * APPROACH_LENGTH
ROBOT_SPAWN_Z = 0.76
WALKABLE_XY_BOUNDS = ((_APPROACH_X_LO, 0.5 * SAND_BED_SIZE[0]), (-0.5 * SAND_BED_SIZE[1], 0.5 * SAND_BED_SIZE[1]))

# Match the particle spawner's cell counts; Newton represents volume as 8 * radius**3.
SAND_LATTICE_RESOLUTION = tuple(max(1, math.ceil(extent / MPM_PARTICLE_SPACING)) for extent in SAND_BED_SIZE)
SAND_LATTICE_CELL_SIZE = tuple(
    (extent / resolution for extent, resolution in zip(SAND_BED_SIZE, SAND_LATTICE_RESOLUTION, strict=True))
)

SAND_PARTICLE_COUNT = math.prod(SAND_LATTICE_RESOLUTION)
MPM_PARTICLE_RADIUS = 0.5 * math.prod(SAND_LATTICE_CELL_SIZE) ** (1.0 / 3.0)
SAND_LOCAL_LOWER = (-0.5 * SAND_BED_SIZE[0], -0.5 * SAND_BED_SIZE[1], SAND_SURFACE_Z - SAND_BED_SIZE[2])
SAND_LOCAL_UPPER = tuple((lower + extent for lower, extent in zip(SAND_LOCAL_LOWER, SAND_BED_SIZE, strict=True)))
SAND_MATERIAL_CFG = MPMParticleMaterialCfg(
    density=1600.0,
    young_modulus=1.0e15,
    poisson_ratio=0.3,
    friction=0.48,
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


@configclass
class G1MPMSceneCfg(InteractiveSceneCfg):
    """G1 on a rigid approach platform leading onto a fixed MPM sand bed."""

    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_G1_29DOF_BOX_FOOT_CFG.init_state.replace(pos=(ROBOT_SPAWN_X, 0.0, ROBOT_SPAWN_Z)),
        spawn=UNITREE_G1_29DOF_BOX_FOOT_CFG.spawn.replace(
            collision_props={
                "/.*_ankle_roll_link/.*": [NewtonCollisionCfg(contact_margin=FOOT_CONTACT_MARGIN, contact_gap=0.0)]
            }
        ),
    )
    sand = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        init_state=MPMObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=MPMGridCfg(
            lower=SAND_LOCAL_LOWER,
            upper=SAND_LOCAL_UPPER,
            voxel_size=MPM_VOXEL_SIZE,
            particles_per_cell=MPM_PARTICLES_PER_CELL,
            particle_placement="cell_center",
            jitter=0.05,
            radius=MPM_PARTICLE_RADIUS,
            material=SAND_MATERIAL_CFG,
            visual_color=MPM_VISUAL_COLOR,
        ),
    )
    bed_floor = _mpm_collider_box(
        "{ENV_REGEX_NS}/MPMBedFloor", size=BED_FLOOR_SIZE, position=BED_FLOOR_POSITION, visible=True
    )
    bed_wall_front = _mpm_collider_box(
        "{ENV_REGEX_NS}/MPMBedWallFront",
        size=BED_WALL_X_SIZE,
        position=(_BED_WALL_X_OFFSET, 0.0, _BED_WALL_CENTER_Z),
        visible=True,
    )
    approach_bank = _mpm_collider_box(
        "{ENV_REGEX_NS}/MPMApproachBank", size=APPROACH_BANK_SIZE, position=APPROACH_BANK_POSITION, visible=False
    )
    bed_wall_left = _mpm_collider_box(
        "{ENV_REGEX_NS}/MPMBedWallLeft",
        size=BED_WALL_Y_SIZE,
        position=(0.0, _BED_WALL_Y_OFFSET, _BED_WALL_CENTER_Z),
        visible=True,
    )
    bed_wall_right = _mpm_collider_box(
        "{ENV_REGEX_NS}/MPMBedWallRight",
        size=BED_WALL_Y_SIZE,
        position=(0.0, -_BED_WALL_Y_OFFSET, _BED_WALL_CENTER_Z),
        visible=True,
    )
    rigid_bed_floor = _rigid_collider_box(
        "{ENV_REGEX_NS}/RigidBedFloor", size=BED_FLOOR_SIZE, position=BED_FLOOR_POSITION
    )
    approach_platform = _rigid_collider_box(
        "{ENV_REGEX_NS}/ApproachPlatform",
        size=APPROACH_PLATFORM_SIZE,
        position=APPROACH_PLATFORM_POSITION,
        visible=False,
    )
    approach_platform_visual = _visual_box(
        "{ENV_REGEX_NS}/ApproachPlatformVisual", size=APPROACH_PLATFORM_SIZE, position=APPROACH_VISUAL_POSITION
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
