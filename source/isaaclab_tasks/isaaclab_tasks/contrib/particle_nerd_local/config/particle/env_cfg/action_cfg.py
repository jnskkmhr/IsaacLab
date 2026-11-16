# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for the `particle` task."""

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.action import NullAction


@configclass
class NullActionCfg(ActionTermCfg):
    """Placeholder action; the scene has nothing to control."""

    class_type = NullAction
    asset_name: str = "anchor"


@configclass
class ParticleActionsCfg:
    """Action specifications for the `particle` environment."""

    null = NullActionCfg()
