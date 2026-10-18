# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for the G1 29-DoF locomotion task among low bars.

Every environment holds :data:`BAR_COUNT` bars: a pair of pillars with a long rod strung between
them at shin height, the obstacle a fallen tree caught between two trunks makes. The rod is a
Newton cable integrated by a VBD solver coupled to the robot's MJWarp solver, so it gives way and
swings back when a foot catches it, which is what the policy has to recover from: retract the swing
foot, reorient, and keep tracking the velocity command.

The rod is not attached to the pillar bodies but welded to the world frame at the points where the
pillars hold it (see :mod:`..spawners`), which is the same constraint as long as the pillars do not
move, and it keeps the whole obstacle clone-safe. The pillars themselves are static colliders, so
they belong to the robot's rigid entry and stop the robot from walking through them.

The bars ring the robot's spawn area, each turned across the radius so that a commanded walk
crosses one whichever way it points. The layout is identical in every environment, since
replication copies one authored environment.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, CableObjectCfg
from isaaclab.utils.configclass import configclass

from ...g1_29dof_rigid.env_cfg.scene_cfg import G1SceneCfg
from ..spawners import AnchoredCableCfg
from .scene_cfg import LOG_BEND_STIFFNESS, LOG_DENSITY, LOG_STRETCH_STIFFNESS

##
# Bar geometry. All values are expressed in the environment frame.
##

BAR_COUNT = 6
"""Number of bars per environment.

Changing it re-generates the scene's rod and pillar assets, so the coupler selectors of
:mod:`.physics_cfg` and the reset event of :mod:`..mdp` follow along without further edits.
"""

BAR_RING_RADIUS = 1.5
"""Distance from the environment origin to the middle of a bar [m].

The bars sit on a ring around the robot's randomized spawn pose, far enough out that the robot
never starts inside one.
"""

BAR_SPAN = 1.4
"""Distance between the two pillars of a bar [m]. Wider than the robot, so it cannot step around."""

BAR_HEIGHT = 0.15
"""Height of the rod above the ground [m].

Mid-shin: high enough that a swing foot catches it instead of clearing it, low enough that the
robot can step over it once it learns to lift the foot.
"""

ROD_RADIUS = 0.04
"""Radius of the rod [m]."""

ROD_SEGMENT_COUNT = 14
"""Number of cable segments in a rod, so a segment is about 0.1 m long."""

PILLAR_SIZE = 0.09
"""Edge length of a pillar's square cross-section [m]."""

PILLAR_MARGIN = 0.05
"""Height a pillar rises above the rod it holds [m]."""

ROD_PRIM_NAME = "Bar"
"""Prefix of the rod prims below each environment, used by the coupler's ownership selectors."""

PILLAR_PRIM_NAME = "Pillar"
"""Prefix of the pillar prims below each environment."""


def _bar_poses() -> list[tuple[float, float, float]]:
    """Spread :data:`BAR_COUNT` bars over a ring around the environment origin.

    Returns:
        The middle [m] and yaw [rad] of every bar, one ``(x, y, yaw)`` triple per bar. The yaw is
        the direction from the origin to the bar, so the rod itself runs across it.
    """
    poses = []
    for bar in range(BAR_COUNT):
        yaw = 2.0 * math.pi * bar / BAR_COUNT
        poses.append((BAR_RING_RADIUS * math.cos(yaw), BAR_RING_RADIUS * math.sin(yaw), yaw))
    return poses


BAR_POSES = _bar_poses()
"""Placement of every bar."""

ROD_ASSET_NAMES = [f"bar_{index}" for index in range(len(BAR_POSES))]
"""Scene names of the rods, used by the reset event of :mod:`..mdp`."""

ROD_ANCHOR_INDICES = (0, 1, ROD_SEGMENT_COUNT - 1, ROD_SEGMENT_COUNT)
"""Control points of a rod welded to the world.

Both points of an end are welded, which clamps the end's orientation the way a rod wedged into a
pillar is held, rather than letting it pivot.
"""


def _rod_control_points(yaw: float) -> list[tuple[float, float, float]]:
    """Build the control points of a rod in its own frame.

    The rod runs across the radius its bar sits on. The yaw is baked into the points rather than
    carried by the cable's initial orientation, so that the anchor sites of :mod:`..spawners`,
    which are authored in the same frame, need no rotation of their own.

    Args:
        yaw: Direction from the environment origin to the bar [rad].

    Returns:
        The control points [m], shape ``(ROD_SEGMENT_COUNT + 1, 3)``.
    """
    points = []
    for index in range(ROD_SEGMENT_COUNT + 1):
        across = (index / ROD_SEGMENT_COUNT - 0.5) * BAR_SPAN
        points.append((-across * math.sin(yaw), across * math.cos(yaw), 0.0))
    return points


def _rod(index: int) -> CableObjectCfg:
    """Build the rod of one bar.

    Args:
        index: Index of the bar in :data:`BAR_POSES`.

    Returns:
        The cable-object configuration of the rod.
    """
    center_x, center_y, yaw = BAR_POSES[index]
    return CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + f"{ROD_PRIM_NAME}_{index}",
        init_state=CableObjectCfg.InitialStateCfg(pos=(center_x, center_y, BAR_HEIGHT)),
        spawn=AnchoredCableCfg(
            positions=_rod_control_points(yaw),
            anchor_indices=ROD_ANCHOR_INDICES,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 0.15), roughness=0.9),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=2.0 * ROD_RADIUS,
                density=LOG_DENSITY,
                stretch_stiffness=LOG_STRETCH_STIFFNESS,
                bend_stiffness=LOG_BEND_STIFFNESS,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
    )


def _pillar(index: int, side: int) -> AssetBaseCfg:
    """Build one pillar of a bar.

    Args:
        index: Index of the bar in :data:`BAR_POSES`.
        side: Which end of the rod the pillar holds, ``0`` or ``1``.

    Returns:
        The asset configuration of the pillar. It carries no rigid body, so it is a static
        collider and belongs to the robot's rigid entry (see :mod:`.physics_cfg`).
    """
    center_x, center_y, yaw = BAR_POSES[index]
    height = BAR_HEIGHT + PILLAR_MARGIN
    along = (side - 0.5) * BAR_SPAN
    offset_x, offset_y = -along * math.sin(yaw), along * math.cos(yaw)
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + f"{PILLAR_PRIM_NAME}_{index}_{side}",
        spawn=sim_utils.CuboidCfg(
            size=(PILLAR_SIZE, PILLAR_SIZE, height),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.22, 0.15), roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(center_x + offset_x, center_y + offset_y, 0.5 * height)),
    )


@configclass
class G1BarSceneCfg(G1SceneCfg):
    """Flat terrain, a G1, and :data:`BAR_COUNT` rods strung between pillars."""

    # A coupled solver keeps its contact forces in per-entry buffers, which Newton contact sensors
    # do not support, so the sensor is dropped and the contact terms are derived from the foot
    # kinematics instead (see :mod:`..mdp`).
    contact_forces = None  # type: ignore

    def __post_init__(self) -> None:
        """Attach one rod and one pillar pair per bar.

        The obstacles are added here rather than declared as fields because their number follows
        :data:`BAR_COUNT`. The scene collects its assets from the instance dictionary, so an
        attribute set here is an asset like any other.
        """
        super().__post_init__()
        for index, name in enumerate(ROD_ASSET_NAMES):
            setattr(self, name, _rod(index))
            for side in (0, 1):
                setattr(self, f"pillar_{index}_{side}", _pillar(index, side))
