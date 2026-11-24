# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.push.config.g1_29dof_mpm_tow.mdp as tow_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp

from .scene_cfg import SLED_ASSET, SLED_MASS

SLED_MASS_START = 5.0
"""Sled mass at the start of training in the environments that follow the curriculum [kg]."""


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    # The sand tow task's schedule: half of the environments start with a 5 kg sled and work up to the full mass.
    sled_mass = CurrTerm(
        func=tow_mdp.sled_mass_levels,
        params={
            "asset_cfg": SceneEntityCfg(SLED_ASSET),
            "mass_range": (SLED_MASS_START, SLED_MASS),
            "mass_step": 1.0,
            "promote_ratio": 0.6,
            "demote_ratio": 0.3,
            "min_commanded_distance": 2.0,
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
    track_lin_vel_weight = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={"term_name": "track_lin_vel_xy", "weight_0": 4.0, "weight_1": 8.0, "step_0": 0, "step_1": 15000 * 24},
    )
