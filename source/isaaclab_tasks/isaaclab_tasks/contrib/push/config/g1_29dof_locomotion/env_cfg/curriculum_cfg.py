# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tracking reward tolerances and actor orientation-bias curricula."""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp


@configclass
class CurriculumCfg:
    """Tighten tracking over 15,000 iterations and ramp orientation bias over 5,000."""

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
    track_base_height = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "base_height",
            "param_name": "std",
            "val_0": 0.2,
            "val_1": 0.05,
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )

    gravity_roll_bias = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.projected_gravity.params.roll_bias_range",
            "modify_fn": mdp.linear_ramp,
            "modify_params": {
                "initial_value": (0.0, 0.0),
                "final_value": (-math.radians(3.0), math.radians(3.0)),
                "start_step": 0,
                "end_step": 5000 * 24,
            },
        },
    )

    gravity_pitch_bias = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.projected_gravity.params.pitch_bias_range",
            "modify_fn": mdp.linear_ramp,
            "modify_params": {
                "initial_value": (0.0, 0.0),
                "final_value": (-math.radians(3.0), math.radians(3.0)),
                "start_step": 0,
                "end_step": 5000 * 24,
            },
        },
    )

    # ang_vel_xy_weight = CurrTerm(
    #     func=g1_mdp.ramp_reward_weight,
    #     params={
    #         "term_name": "ang_vel_xy",
    #         "weight_0": -0.25,
    #         "weight_1": -1.0,
    #         "step_0": 0,
    #         "step_1": 10000 * 24,
    #     }
    # )
