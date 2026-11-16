# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

from particle_nerd_benchmark.core.mdp.actions import ObjectStateAction


@configclass
class ActionsCfg:
    motion = ActionTermCfg(class_type=ObjectStateAction, asset_name="object")
