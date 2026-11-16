# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for the `sphere` two-way coupling task."""

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.observation import SphereWrenchAction


@configclass
class SphereWrenchActionCfg(ActionTermCfg):
    """6D world-frame wrench action applied to the rigid sphere center of mass."""

    class_type = SphereWrenchAction
    asset_name: str = "sphere"
    scale: float = 250.0


@configclass
class SphereActionsCfg:
    """Action specifications for the `sphere` two-way coupling environment."""

    wrench = SphereWrenchActionCfg()
