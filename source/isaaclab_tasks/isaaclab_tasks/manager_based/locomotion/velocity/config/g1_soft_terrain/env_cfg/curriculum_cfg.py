# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_soft_terrain.mdp as g1_mdp

@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_stiffness = CurrTerm(
        func=g1_mdp.update_terrain_stiffness, 
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_solver_name": "physics_callback",
            },
        )
    terrain_ground_level = CurrTerm(
        func=g1_mdp.terrain_ground_level,
        params={
            "rl_horizon": 24,
            "max_iterations": 1500,
            "minimum_ground_height": -0.1,
            },
        )