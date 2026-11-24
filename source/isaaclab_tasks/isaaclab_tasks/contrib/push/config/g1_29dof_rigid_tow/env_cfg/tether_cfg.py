# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The rope between the robot and the sled."""

from isaaclab.utils import configclass

from .scene_cfg import SLED_ASSET, SLED_SIZE


@configclass
class TetherCfg:
    """A rope that pulls only when taut, modelled as a one-sided spring and damper between two points."""

    robot_body: str = "pelvis"
    """Robot body the rope is tied to."""

    robot_offset: tuple[float, float, float] = (-0.10, 0.0, 0.0)
    """Attachment point in the robot body's frame [m]: the back of the waist."""

    object_name: str = SLED_ASSET
    """Scene name of the towed object."""

    object_offset: tuple[float, float, float] = (0.5 * SLED_SIZE[0], 0.0, -0.5 * SLED_SIZE[2] + 0.05)
    """Attachment point in the object's frame [m]: the middle of the front face, 0.05 m above the bottom."""

    length: float = 1.2
    """Rope length [m]; the rope carries no force while the two points are closer than this."""

    stiffness: float = 1000.0
    """Rope stiffness once taut [N/m]."""

    damping: float = 40.0
    """Damping of the stretch rate once taut [N s/m]."""

    max_force: float = 300.0
    """Upper limit of the rope tension [N]."""
