# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.vel_mdp import contact_terrain as contact_mdp


@configclass
class G1CurriculumCfg:
    """Reference moving-patch curriculum with rigid-terrain progression."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    command_vel = CurrTerm(
        func=g1_mdp.commands_vel,  # type: ignore
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 10000 * 24, "lin_vel_x": (-1.0, 1.7), "ang_vel_z": (-0.7, 0.7)},
                {"step": 15000 * 24, "lin_vel_x": (-1.0, 2.5), "ang_vel_z": (-1.0, 1.0)},
            ],
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
        params={
            "term_name": "track_lin_vel_xy",
            "weight_0": 4.0,
            "weight_1": 8.0,
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )


@configclass
class G1RoughCurriculumCfg:
    """Task-local rough-terrain configuration."""

    terrain_levels = CurrTerm(func=contact_mdp.terrain_levels_vel)
    command_vel = CurrTerm(
        func=g1_mdp.commands_vel,
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 240000, "lin_vel_x": (-1.0, 1.7), "ang_vel_z": (-0.7, 0.7)},
                {"step": 360000, "lin_vel_x": (-1.0, 2.5), "ang_vel_z": (-1.0, 1.0)},
            ],
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
                "end_step": 120000,
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
                "end_step": 120000,
            },
        },
    )
    track_lin_vel = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_lin_vel_xy",
            "param_name": "std",
            "val_0": 1.0,
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 360000,
        },
    )
    track_ang_vel = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_ang_vel_z",
            "param_name": "std",
            "val_0": 1.0,
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 360000,
        },
    )
    track_heading = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_heading",
            "param_name": "std",
            "val_0": 1.0,
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 360000,
        },
    )
    track_lin_vel_weight = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={"term_name": "track_lin_vel_xy", "weight_0": 4.0, "weight_1": 8.0, "step_0": 0, "step_1": 360000},
    )


@configclass
class G1RoughCurriculumCfg_PLAY(G1RoughCurriculumCfg):
    """Playback configuration."""

    gravity_roll_bias = None
    gravity_pitch_bias = None
