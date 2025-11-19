# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(vel_mdp.lin_vel_cmd_levels)