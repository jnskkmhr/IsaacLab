# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for joint-limit-derived position target scaling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .joint_actions import JointPositionRangeAction


@configclass
class JointPositionRangeActionCfg(JointPositionActionCfg):
    """Position targets centered on the default pose with symmetric per-joint ranges.

    ``scale`` multiplies the distance from the default position to the farther
    physical joint limit. Use 1.0 for that full range, or larger values to amplify
    it. This does not clamp targets to joint limits; ``clip`` remains optional.
    ``use_default_offset`` must remain True.
    """

    class_type: type[JointPositionRangeAction] | str = "{DIR}.joint_actions:JointPositionRangeAction"
