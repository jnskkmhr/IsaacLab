# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for the `snow_ball` slope rolling task."""

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

from ..mdp import SnowBallWrenchAction


@configclass
class SnowBallActionsCfg:
    """6-DoF wrench action applied to the rolling snow-ball collider."""

    wrench = ActionTermCfg(class_type=SnowBallWrenchAction, asset_name="snow_ball")
