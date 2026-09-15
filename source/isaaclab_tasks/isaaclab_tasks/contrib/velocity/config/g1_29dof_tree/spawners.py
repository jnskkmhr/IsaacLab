# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawner for a cable whose end points are held in place.

Newton lowers an AOUSD ``PhysicsAttachment`` prim with a hard (unauthored) stiffness onto a ball
joint between a cable point and the target xform's body, and an xform that belongs to no rigid body
resolves to the world frame (see ``newton/_src/utils/import_usd_deformable_attachments.py``). The
attachment and the xform it targets are authored *below* the cable prim, because environment
replication copies one asset subtree at a time and only rewrites relationship targets that live
inside the copied subtree; an anchor outside it would leave every environment's cable welded to the
first environment's anchor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd

ANCHOR_FRAME_NAME = "anchor_frame"
"""Name of the xform the anchor joints are defined against, relative to the cable's geometry."""

ANCHOR_PRIM_NAME = "anchor"
"""Name of the ``PhysicsAttachment`` prim, relative to the cable prim."""


@configclass
class AnchoredCableCfg(sim_utils.CableCfg):
    """A cable with some of its control points welded to the world frame.

    Welding a single point leaves the cable free to swing about it; welding the two points at an
    end clamps the end's orientation as well, so the cable behaves like a rod built into a support.
    """

    func: str = "{DIR}.spawners:spawn_anchored_cable"

    anchor_indices: Sequence[int] = MISSING
    """Indices into :attr:`~isaaclab.sim.spawners.shapes.CableCfg.positions` welded to the world."""


def spawn_anchored_cable(
    prim_path: str,
    cfg: AnchoredCableCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a cable and weld its :attr:`AnchoredCableCfg.anchor_indices` to the world frame.

    Args:
        prim_path: The prim path or pattern to spawn the cable at.
        cfg: The anchored-cable configuration.
        translation: Local translation of the cable root [m].
        orientation: Local orientation in ``(x, y, z, w)`` order.
        **kwargs: Additional arguments consumed by the cable spawner.

    Returns:
        The created cable root prim.

    Raises:
        ValueError: If an anchor index does not address a control point.
    """
    from pxr import Sdf

    if not len(cfg.anchor_indices):
        raise ValueError("AnchoredCableCfg requires at least one anchor index.")
    if any(not 0 <= index < len(cfg.positions) for index in cfg.anchor_indices):
        raise ValueError(
            f"AnchoredCableCfg anchor indices {list(cfg.anchor_indices)} must address one of the "
            f"{len(cfg.positions)} control points."
        )

    prim = sim_utils.spawn_cable(prim_path, cfg, translation, orientation, **kwargs)

    stage = get_current_stage()
    for path in find_matching_prim_paths(prim_path):
        # The anchor shares the curve's frame, so an anchor site is simply its control point.
        anchor_path = f"{path}/geometry/{ANCHOR_FRAME_NAME}"
        stage.DefinePrim(anchor_path, "Xform")
        attachment = stage.DefinePrim(f"{path}/{ANCHOR_PRIM_NAME}", "PhysicsAttachment")
        attachment.CreateRelationship("physics:src0").SetTargets([f"{path}/geometry/mesh"])
        attachment.CreateRelationship("physics:src1").SetTargets([anchor_path])
        attachment.CreateAttribute("physics:type0", Sdf.ValueTypeNames.Token).Set("point")
        attachment.CreateAttribute("physics:type1", Sdf.ValueTypeNames.Token).Set("xform")
        attachment.CreateAttribute("physics:indices0", Sdf.ValueTypeNames.IntArray).Set(
            [int(index) for index in cfg.anchor_indices]
        )
        attachment.CreateAttribute("physics:coords1", Sdf.ValueTypeNames.Vector3fArray).Set(
            [tuple(cfg.positions[index]) for index in cfg.anchor_indices]
        )
        # Stiffness is left unauthored: only an infinitely stiff attachment lowers to a joint.

    return prim
