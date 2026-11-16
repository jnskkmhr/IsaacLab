# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from particle_nerd_benchmark.core.mdp.events import reset_sand


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=reset_scene_to_default, mode="reset")
    reset_sand = EventTermCfg(func=reset_sand, mode="reset")
