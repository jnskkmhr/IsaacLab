# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for the G1 29-DoF locomotion task on a Newton MPM granular bed.

No contact sensor is declared: coupled solvers keep contact forces in per-entry buffers, which
Newton contact sensors do not support. Foot contact is derived from the particle state instead,
see :class:`~isaaclab_tasks.contrib.g1_29dof_mpm.g1_mpm_env.G1MPMEnv`.

The bed is authored so that its free surface lies at ``z = 0``. Every flat-ground assumption
carried over from the rigid and soft-contact G1 tasks (base-height target, foot clearance
reference, root-height termination) therefore stays valid without a terrain importer.
"""

import math

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

##
# Pre-defined configs
##
from isaaclab_assets import UNITREE_G1_29DOF_CFG

##
# Granular bed geometry. All values are expressed in the environment frame.
##

MPM_VOXEL_SIZE = 0.05
"""Background-grid voxel size of the implicit MPM solver [m]."""

MPM_COLLIDER_MARGIN = 0.5 * MPM_VOXEL_SIZE
"""Contact margin of the shapes owned by the MPM entry [m]."""

FOOT_CONTACT_MARGIN = 0.75 * MPM_VOXEL_SIZE
"""Outward inflation of the robot collision surfaces seen by the granular solver [m].

Implicit MPM resolves a collider through how much of a background voxel it occupies, so a sole
thinner than one voxel blocks almost no cell and the bed carries almost no load. Inflating the
robot shapes is what turns the feet into load-bearing colliders; with the default zero margin the
robot sinks straight through the bed.

The fraction is a stability limit, not a modelling choice. A full-voxel margin supports the robot
at its nominal standing height but traps particles inside the sole, and a coupled entry cannot
project them back out (:attr:`MPMSolverCfg.project_outside_colliders` is rejected there), so a
world diverges into ``NaN`` every few hundred steps. Three quarters of a voxel holds the robot at
roughly 5 cm of sinkage and ran clean over 600 steps across four worlds.

The rigid entry only ever meets the hidden catch-net pan below the bed, so the same inflation is
inconsequential there.
"""

MPM_PARTICLE_SPACING = 0.04
"""Lattice spacing requested when sampling the bed [m].

Implicit MPM needs roughly two particles per background voxel to transfer stress; a bed sampled
at or above :attr:`MPM_VOXEL_SIZE` carries no load and blows up within a few steps. The ratio
here matches the standalone Newton G1 example (``0.025`` voxels sampled at ``0.02``).
"""

# MPM_VISUAL_COLOR = (0.45, 0.40, 0.28)
MPM_VISUAL_COLOR = (0.32, 0.24, 0.21)
"""Display color of the granular bed."""

SAND_BED_SIZE = (3.0, 3.0, 0.25)
"""Extent of the deformable bed [m], ``(x, y, z)``.

The depth spans several background voxels so that a loaded foot reaches bearing capacity inside
the granular column instead of bottoming out on the rigid pan.
"""

SAND_SURFACE_Z = 0.0
"""World height of the undisturbed bed surface [m]."""

BED_WALL_THICKNESS = 0.1
"""Thickness of the retaining walls that keep spilled particles inside the bed [m]."""

BED_WALL_HEIGHT = 0.4
"""Height of the retaining walls [m]."""

BED_FLOOR_THICKNESS = 0.1
"""Thickness of the rigid pan the bed rests on [m]."""

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
"""Horizontal bed extent in the environment frame [m], ``((x_lo, x_hi), (y_lo, y_hi))``."""

##
# Particle lattice derived from the bed geometry.
##

SAND_LATTICE_RESOLUTION = tuple(max(1, math.ceil(extent / MPM_PARTICLE_SPACING)) for extent in SAND_BED_SIZE)
SAND_LATTICE_CELL_SIZE = tuple(
    extent / resolution for extent, resolution in zip(SAND_BED_SIZE, SAND_LATTICE_RESOLUTION, strict=True)
)
SAND_PARTICLE_COUNT = math.prod(SAND_LATTICE_RESOLUTION)
"""Number of particles generated per environment."""

MPM_PARTICLE_RADIUS = 0.5 * math.prod(SAND_LATTICE_CELL_SIZE) ** (1.0 / 3.0)
"""Equal-volume particle radius [m]; Newton represents a particle as a ``8 * radius**3`` cube."""

SAND_LOCAL_LOWER = (
    -0.5 * SAND_BED_SIZE[0],
    -0.5 * SAND_BED_SIZE[1],
    SAND_SURFACE_Z - SAND_BED_SIZE[2],
)
SAND_LOCAL_UPPER = tuple(lower + extent for lower, extent in zip(SAND_LOCAL_LOWER, SAND_BED_SIZE, strict=True))

# Dry sand, matching the standalone Newton G1 example: 1500 kg/m^3 bulk density at 0.6 packing
# density and a 30 deg internal friction angle.
SAND_MATERIAL_CFG = MPMParticleMaterialCfg(
    density=1500.0 * 0.6,
    young_modulus=15.0e6,
    poisson_ratio=0.3,
    friction=math.tan(math.radians(30.0)),
    yield_pressure=1.0e12,
)


def _mpm_collider_box(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    visible: bool = False,
) -> RigidObjectCfg:
    """Build a kinematic box owned exclusively by the MPM entry.

    Args:
        prim_path: Prim path of the collider.
        size: Box extent [m], shape ``(3,)``.
        position: Box center in the environment frame [m], shape ``(3,)``.
        visible: Whether the box is rendered.

    Returns:
        The rigid-object configuration of the collider.
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=MPM_COLLIDER_MARGIN,
                # Implicit MPM consumes the shape margin; gap is a rigid-contact parameter.
                contact_gap=0.0,
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25), roughness=0.8),
            visible=visible,
        ),
    )


def _rigid_collider_box(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
) -> AssetBaseCfg:
    """Build the hidden static mirror of an MPM collider used by the rigid subsolver.

    Args:
        prim_path: Prim path of the mirror.
        size: Box extent [m], shape ``(3,)``.
        position: Box center in the environment frame [m], shape ``(3,)``.

    Returns:
        The asset configuration of the mirror.
    """
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=0.004,
                contact_gap=0.002,
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
            visible=False,
        ),
    )


@configclass
class G1MPMSceneCfg(InteractiveSceneCfg):
    """G1 standing on a per-environment MPM bed held by a rigid pan."""

    # robots
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UNITREE_G1_29DOF_CFG.spawn.replace( # type: ignore
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=FOOT_CONTACT_MARGIN,
                # Implicit MPM consumes the shape margin; gap is a rigid-contact parameter.
                contact_gap=0.0,
            )
        ),
    )

    # granular terrain
    sand = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        init_state=MPMObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=MPMGridCfg(
            lower=SAND_LOCAL_LOWER,
            upper=SAND_LOCAL_UPPER,
            voxel_size=MPM_PARTICLE_SPACING,
            particles_per_cell=1.0,
            particle_placement="cell_center",
            # A fresh arrangement is drawn per reset instead of repeating one build-time sample.
            jitter=0.0,
            radius=MPM_PARTICLE_RADIUS,
            material=SAND_MATERIAL_CFG,
            visual_color=MPM_VISUAL_COLOR,
        ),
    )

    # MPM-owned pan. Particles that reach the walls are retained instead of leaving the solver.
    bed_floor = _mpm_collider_box(
        "{ENV_REGEX_NS}/MPMBedFloor",
        size=BED_FLOOR_SIZE,
        position=BED_FLOOR_POSITION,
        visible=True,
    )
    bed_wall_front = _mpm_collider_box(
        "{ENV_REGEX_NS}/MPMBedWallFront",
        size=BED_WALL_X_SIZE,
        position=(_BED_WALL_X_OFFSET, 0.0, _BED_WALL_CENTER_Z),
        visible=True,
    )
    bed_wall_back = _mpm_collider_box(
        "{ENV_REGEX_NS}/MPMBedWallBack",
        size=BED_WALL_X_SIZE,
        position=(-_BED_WALL_X_OFFSET, 0.0, _BED_WALL_CENTER_Z),
        visible=True,
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

    # Static mirror of the pan floor. It belongs to the rigid entry and catches a robot that has
    # sunk through the whole bed, so a failed episode terminates instead of falling forever.
    rigid_bed_floor = _rigid_collider_box(
        "{ENV_REGEX_NS}/RigidBedFloor",
        size=BED_FLOOR_SIZE,
        position=BED_FLOOR_POSITION,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
