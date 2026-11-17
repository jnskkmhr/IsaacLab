# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for the G1 29-DoF locomotion task among fallen trees.

Every environment holds :data:`PILE_COUNT` piles of logs the robot has to walk through. A log is a
Newton cable: a chain of capsule bodies joined by stretch, shear, bend and twist constraints,
integrated by a VBD solver, while the robot is integrated by MJWarp and the two meet through the
lagged proxy coupling of :mod:`.physics_cfg`.

A pile is laid out the way ``newton/examples/cable/example_cable_pile.py`` lays out its pile: lanes
of cables stacked in layers whose orientation alternates between the x and the y axis, each cable
given a sinusoidal waviness so that no two of them rest flush against one another. The logs start
just above their resting height and settle within the first few steps. The piles themselves sit on
a ring around the robot's spawn area, each rotated to face outwards, so that the robot meets a pile
whichever way it is commanded to walk.

The layout is identical in every environment, since replication copies one authored environment and
a per-environment layout would mean giving up ``replicate_physics``. Variety comes from the
randomized spawn pose of the robot and from which pile the command drives it into.
"""

from __future__ import annotations

import math

from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObjectCfg, RigidObjectCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

from ...g1_29dof_rigid.env_cfg.scene_cfg import G1SceneCfg

##
# Log geometry. All values are expressed in the environment frame.
##

LOG_LENGTH = 1.2
"""Length of a log [m]. Long enough to trip the robot, short enough to be kicked aside."""

LOG_SEGMENT_COUNT = 12
"""Number of cable segments per log, so a segment is 0.1 m long."""

LOG_RADIUS = 0.02
"""Radius of a log [m]."""

LOG_DENSITY = 600.0
"""Log density [kg/m^3], in the range of green softwood. A log weighs about 2 kg."""

# LOG_STRETCH_STIFFNESS = 1.77e8
LOG_STRETCH_STIFFNESS = 1.0e6
"""Stretch modulus of a log [Pa].

Isaac Lab authors moduli, which Newton's cable importer multiplies by the cross-section area to get
the rod stiffness. At this radius the value reproduces the 5e5 N stretch stiffness of the standalone
``example_cable_pile.py``.
"""

# LOG_BEND_STIFFNESS = 1.57e8
LOG_BEND_STIFFNESS = 1.0e5
"""Bend modulus of a log [Pa].

The importer multiplies it by the second moment of area, which at this radius reproduces the
100 N*m^2 bend stiffness of ``example_cable_pile.py``: a log sags visibly across a gap but keeps
its shape while it is pushed around.
"""

##
# Pile layout.
##

PILE_COUNT = 8
"""Number of log piles per environment.

Changing it re-generates the scene's log assets, so the coupler selectors of :mod:`.physics_cfg`
and the reset event of :mod:`..mdp` follow along without further edits. Each pile costs
:data:`PILE_LAYERS` * :data:`PILE_LANES` cables per environment, which is the dominant cost of the
task, so raise it together with a lower environment count.
"""

PILE_RING_RADIUS = 1.5
"""Distance from the environment origin to the center of a pile [m].

The piles sit on a ring around the robot's randomized spawn pose, far enough out that the robot
never starts inside one and close enough that a commanded walk of about a second reaches one.
"""

PILE_LAYERS = 3
"""Number of stacked layers in a pile."""

PILE_LANES = 3
"""Number of logs per layer."""

LANE_SPACING = 0.24
"""Distance between the lanes of a layer [m]. Eight log radii, as in ``example_cable_pile.py``."""

LAYER_GAP = 0.07
"""Vertical distance between layers [m]. Slightly more than a log diameter, so the logs settle."""

WAVINESS_AMPLITUDE = 0.03
"""Amplitude of the sinusoidal waviness of a log [m], as in ``example_cable_pile.py``."""

WAVINESS_CYCLES = 2.0
"""Number of waviness periods along a log."""

GROUND_SINK = 0.01
"""Depth the cable-entry ground is placed below the terrain [m].

The piles rest on a ground of their own, because the terrain plane is a static shape and belongs to
the rigid entry (see :mod:`.physics_cfg`). Sinking it keeps the robot's proxies inside the cable
entry clear of it, so the robot is not held up by two floors at once. The logs therefore lie a
centimetre deeper than the terrain they are drawn against.
"""

PILE_GROUND_SIZE = (5.0, 5.0, 0.2)
"""Extent of the cable-entry ground box [m], shape ``(3,)``.

Wide enough to hold every pile on the ring, plus room for kicked-away logs.
"""

LOG_PRIM_NAME = "Log"
"""Prefix of the log prims below each environment, used by the coupler's ownership selectors."""

PILE_GROUND_PRIM_NAME = "PileGround"
"""Name of the cable-entry ground prim below each environment."""


def _rotate(x: float, y: float, yaw: float) -> tuple[float, float]:
    """Rotate a point of the horizontal plane [m] about the origin by ``yaw`` [rad]."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    return x * cos - y * sin, x * sin + y * cos


def _pile_poses() -> list[tuple[float, float, float]]:
    """Spread :data:`PILE_COUNT` piles over a ring around the environment origin.

    Returns:
        The center [m] and yaw [rad] of every pile, one ``(x, y, yaw)`` triple per pile. A single
        pile sits straight ahead of the robot.
    """
    poses = []
    for pile in range(PILE_COUNT):
        yaw = 2.0 * math.pi * pile / PILE_COUNT
        poses.append((PILE_RING_RADIUS * math.cos(yaw), PILE_RING_RADIUS * math.sin(yaw), yaw))
    return poses


def _log_control_points(axis: int, yaw: float) -> list[tuple[float, float, float]]:
    """Build the control points of one log in its own frame.

    Args:
        axis: Axis the log runs along in its pile's frame, ``0`` for x and ``1`` for y.
        yaw: Yaw of the pile the log belongs to [rad].

    Returns:
        The control points [m], shape ``(LOG_SEGMENT_COUNT + 1, 3)``.
    """
    points = []
    for index in range(LOG_SEGMENT_COUNT + 1):
        fraction = index / LOG_SEGMENT_COUNT
        along = (fraction - 0.5) * LOG_LENGTH
        across = WAVINESS_AMPLITUDE * math.sin(2.0 * math.pi * WAVINESS_CYCLES * fraction)
        x, y = (along, across) if axis == 0 else (across, along)
        points.append((*_rotate(x, y, yaw), 0.0))
    return points


def _log_placements() -> list[tuple[tuple[float, float, float], int, float]]:
    """Build the layout of every pile: one placement per log.

    Within a pile, layers alternate between logs running along x and logs running along y, and the
    lanes of a layer are spread along the other axis; the whole pile is then rotated and moved onto
    its place on the ring.

    Returns:
        The log position in the environment frame [m], its axis, and its pile's yaw [rad], one
        triple per log.
    """
    placements = []
    for center_x, center_y, yaw in _pile_poses():
        for layer in range(PILE_LAYERS):
            axis = layer % 2
            height = LOG_RADIUS + 0.005 + layer * LAYER_GAP
            for lane in range(PILE_LANES):
                offset = (lane - (PILE_LANES - 1) * 0.5) * LANE_SPACING
                local_x, local_y = (0.0, offset) if axis == 0 else (offset, 0.0)
                x, y = _rotate(local_x, local_y, yaw)
                placements.append(((center_x + x, center_y + y, height), axis, yaw))
    return placements


LOG_PLACEMENTS = _log_placements()
"""Placement of every log of every pile."""

LOG_ASSET_NAMES = [f"log_{index}" for index in range(len(LOG_PLACEMENTS))]
"""Scene names of the logs, used by the reset event of :mod:`..mdp`."""


def _log(index: int) -> CableObjectCfg:
    """Build one log of a pile.

    Args:
        index: Index of the log in :data:`LOG_PLACEMENTS`.

    Returns:
        The cable-object configuration of the log.
    """
    position, axis, yaw = LOG_PLACEMENTS[index]
    return CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + f"{LOG_PRIM_NAME}_{index}",
        init_state=CableObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CableCfg(
            positions=_log_control_points(axis, yaw),
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 0.15), roughness=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(161 / 255, 118 / 255, 66 / 255), roughness=0.9),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=2.0 * LOG_RADIUS,
                density=LOG_DENSITY,
                stretch_stiffness=LOG_STRETCH_STIFFNESS,
                bend_stiffness=LOG_BEND_STIFFNESS,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
    )


def _pile_ground() -> RigidObjectCfg:
    """Build the kinematic ground the piles rest on, owned by the cable entry.

    Returns:
        The rigid-object configuration of the ground box.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + PILE_GROUND_PRIM_NAME,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -GROUND_SINK - 0.5 * PILE_GROUND_SIZE[2])),
        spawn=sim_utils.CuboidCfg(
            size=PILE_GROUND_SIZE,
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                # No inflation: the clearance to the robot's own floor is what keeps the robot's
                # proxies off this box, and a margin would eat into it.
                contact_margin=0.0,
                contact_gap=0.0,
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
            # The box only exists for the cable solver; the terrain is what the scene shows.
            visible=False,
        ),
    )


@configclass
class G1TreeSceneCfg(G1SceneCfg):
    """Flat terrain, a G1, and :data:`PILE_COUNT` piles of fallen trees."""

    # A coupled solver keeps its contact forces in per-entry buffers, which Newton contact sensors
    # do not support, so the sensor is dropped and the contact terms are derived from the foot
    # kinematics instead (see :mod:`..mdp`).
    contact_forces = None  # type: ignore

    pile_ground: RigidObjectCfg = _pile_ground()

    def __post_init__(self) -> None:
        """Attach one cable asset per log.

        The logs are added here rather than declared as fields because their number follows
        :data:`PILE_COUNT`. The scene collects its assets from the instance dictionary, so an
        attribute set here is an asset like any other.
        """
        super().__post_init__()
        for index, name in enumerate(LOG_ASSET_NAMES):
            setattr(self, name, _log(index))
