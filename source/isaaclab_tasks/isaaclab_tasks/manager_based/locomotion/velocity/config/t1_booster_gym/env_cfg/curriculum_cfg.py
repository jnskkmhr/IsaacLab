# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_booster_gym.mdp as t1_mdp

@configclass
class T1CurriculumsCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)

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

    action_rate_currilcum = CurrTerm(
        func=t1_mdp.linearly_alter_weight,
        params={
            "term_name": "action_rate_l2",
            "start_weight": -0.0005,
            "end_weight": -0.01,
            "start_step": 4000 * 24,
            "end_step": 8000 * 24
        }
    )

    command_curriculum_x = CurrTerm(
        func=t1_mdp.linear_alter_command_param,
        params={
            "term_name": "base_velocity",
            "subterm_name": "lin_vel_x",
            "start_step": 300 * 24,
            "end_step": 3000 * 24,
            "start_value": 0.7,
            "end_value": 1.2
        }
    )
    command_curriculum_y = CurrTerm(
        func=t1_mdp.linear_alter_command_param,
        params={
            "term_name": "base_velocity",
            "subterm_name": "lin_vel_y",
            "start_step": 300 * 24,
            "end_step": 3000 * 24,
            "start_value": 0.3,
            "end_value": 0.6
        }
    )
    command_curriculum_z = CurrTerm(
        func=t1_mdp.linear_alter_command_param,
        params={
            "term_name": "base_velocity",
            "subterm_name": "ang_vel_z",
            "start_step": 600 * 24,
            "end_step": 3500 * 24,
            "start_value": 0.5,
            "end_value": 2.0
        }
    )

    push_curriculum = CurrTerm(
        func=t1_mdp.modify_event_on_greater_than_reward,
        params={
            "term_name": "push_robot",
            "term_param_name": "velocity_range",
            "value": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-0.2, 0.2),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.4, 0.4)
            },
            "metric_name": "survival",
            "metric_threshold": 1.0 * 0.80,
        }  # on first time reward > 75% survival
    )