# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as Term
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid import mdp as rigid_mdp


@configclass
class G1CurriculumCfg:
    """Command, terrain and reward ramps saved in tyiby5k6."""

    command_vel = Term(
        func=rigid_mdp.commands_vel,
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "ang_vel_z": [-0.5, 0.5], "lin_vel_x": [-1, 1]},
                {"step": 240000, "ang_vel_z": [-0.7, 0.7], "lin_vel_x": [-1, 1.7]},
                {"step": 360000, "ang_vel_z": [-1, 1], "lin_vel_x": [-1, 2.5]},
            ],
        },
    )

    terrain_levels = Term(func=rigid_mdp.terrain_levels_vel, params={})

    track_ang_vel_std = Term(
        func=rigid_mdp.ramp_reward_param,
        params={
            "val_0": 1,
            "val_1": 0.31622776601683794,
            "step_0": 0,
            "step_1": 360000,
            "term_name": "track_ang_vel_z",
            "param_name": "std",
        },
    )

    track_heading_std = Term(
        func=rigid_mdp.ramp_reward_param,
        params={
            "val_0": 1,
            "val_1": 0.31622776601683794,
            "step_0": 0,
            "step_1": 360000,
            "term_name": "track_heading",
            "param_name": "std",
        },
    )

    track_lin_vel_std = Term(
        func=rigid_mdp.ramp_reward_param,
        params={
            "val_0": 1,
            "val_1": 0.31622776601683794,
            "step_0": 0,
            "step_1": 360000,
            "term_name": "track_lin_vel_xy",
            "param_name": "std",
        },
    )

    track_heading_weight = Term(
        func=rigid_mdp.ramp_reward_weight,
        params={"step_0": 0, "step_1": 360000, "weight_0": 4, "weight_1": 8, "term_name": "track_heading"},
    )

    track_lin_vel_weight = Term(
        func=rigid_mdp.ramp_reward_weight,
        params={"step_0": 0, "step_1": 360000, "weight_0": 4, "weight_1": 8, "term_name": "track_lin_vel_xy"},
    )
