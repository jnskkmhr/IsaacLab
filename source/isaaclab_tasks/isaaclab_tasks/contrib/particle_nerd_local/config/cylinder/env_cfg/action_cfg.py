# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for the `cylinder` two-way coupling task."""

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.observation import CylinderWrenchAction


@configclass
class CylinderWrenchActionCfg(ActionTermCfg):
    """6D world-frame wrench action applied to the rigid cylinder center of mass."""

    class_type = CylinderWrenchAction
    asset_name: str = "cylinder"
    scale: float = 250.0


@configclass
class CylinderActionsCfg:
    """Action specifications for the `cylinder` two-way coupling environment."""

    wrench = CylinderWrenchActionCfg()
