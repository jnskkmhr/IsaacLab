# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    # No command-speed ramp: the finite bed is only a few metres long.

    track_lin_vel = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_lin_vel_xy",
            "param_name": "std",
            "val_0": math.sqrt(1.0),
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )

    track_ang_vel = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_ang_vel_z",
            "param_name": "std",
            "val_0": math.sqrt(1.0),
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )

    track_heading = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_heading",
            "param_name": "std",
            "val_0": math.sqrt(1.0),
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )

    # weight curriculum
    track_lin_vel_weight = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={
            "term_name": "track_lin_vel_xy",
            "weight_0": 4.0,
            "weight_1": 8.0,
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )
