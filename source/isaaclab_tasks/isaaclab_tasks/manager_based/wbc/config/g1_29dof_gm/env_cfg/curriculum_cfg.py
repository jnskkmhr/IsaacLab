# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.wbc.mdp as wbc_mdp


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    terrain_levels = CurrTerm(
        func=wbc_mdp.terrain_levels_motion_success,
        params={"command_name": "motion"}
    )