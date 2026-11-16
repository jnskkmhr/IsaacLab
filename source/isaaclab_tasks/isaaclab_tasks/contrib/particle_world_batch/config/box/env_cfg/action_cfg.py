# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for the `box` two-way coupling task."""

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.observation import BoxWrenchAction


@configclass
class BoxActionsCfg:
    """6-DoF wrench action applied to the rigid box center of mass."""

    wrench = ActionTermCfg(class_type=BoxWrenchAction, asset_name="box")
