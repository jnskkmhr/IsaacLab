# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event configuration for the `snow_ball` slope rolling task."""

from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from ..mdp import reset_snow_particles


@configclass
class SnowBallEventsCfg:
    """Reset both the rolling snow-ball collider and the snow pack particle bed."""

    reset_scene = EventTermCfg(func=reset_scene_to_default, mode="reset")
    reset_sand = EventTermCfg(func=reset_snow_particles, mode="reset")
