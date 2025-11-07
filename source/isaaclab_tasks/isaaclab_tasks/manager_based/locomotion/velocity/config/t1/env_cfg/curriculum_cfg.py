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
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1.mdp as t1_mdp

@configclass
class T1CurriculumsCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # command_range = CurrTerm(
    #     func=t1_mdp.modify_command_range, # type: ignore
    #     params={
    #         "term_name": "base_velocity",
    #         "ranges": mdp.UniformVelocityCommandCfg.Ranges( # type: ignore
    #         lin_vel_x=(-1.0, 1.0),
    #         lin_vel_y=(-1.0, 1.0),
    #         ang_vel_z=(-1.0, 1.0)
    #         ),
    #         "num_steps": 10000*24,
    #     },
    # )

    track_lin_vel = CurrTerm(
        func=t1_mdp.modify_reward_std,  # type: ignore
        params={"term_name": "track_lin_vel_xy_exp", "std": 0.25, "num_steps": 10000 * 24}
    )

    track_ang_vel = CurrTerm(
        func=t1_mdp.modify_reward_std,  # type: ignore
        params={"term_name": "track_ang_vel_z_exp", "std": 0.25, "num_steps": 10000 * 24}
    )