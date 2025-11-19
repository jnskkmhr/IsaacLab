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
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_soft.mdp as t1_mdp

@configclass
class T1CurriculumsCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)
    # terrain_levels_max = CurrTerm(func=vel_mdp.terrain_levels_vel_max)

    track_lin_vel = CurrTerm(
        func=t1_mdp.modify_reward_std,  # type: ignore
        params={"term_name": "track_lin_vel_xy_exp", "std": 0.25, "num_steps": 7000 * 24}
    )

    track_ang_vel = CurrTerm(
        func=t1_mdp.modify_reward_std,  # type: ignore
        params={"term_name": "track_ang_vel_z_exp", "std": 0.25, "num_steps": 7000 * 24}
    )