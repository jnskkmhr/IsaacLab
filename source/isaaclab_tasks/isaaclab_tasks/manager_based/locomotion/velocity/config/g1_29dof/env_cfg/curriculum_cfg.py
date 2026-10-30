# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)  # type: ignore
    command_vel = CurrTerm(
        func=vel_mdp.commands_vel,  # type: ignore
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 10000 * 24, "lin_vel_x": (-1.0, 1.7), "ang_vel_z": (-0.7, 0.7)},
                {"step": 15000 * 24, "lin_vel_x": (-1.0, 2.5), "ang_vel_z": (-1.0, 1.0)},
                # {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                # {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "ang_vel_z": (-0.7, 0.7)},
                # {"step": 10000 * 24, "lin_vel_x": (-1.0, 3.0), "ang_vel_z": (-1.0, 1.0)},
            ],
        },
    )

    track_lin_vel = CurrTerm(
        func=vel_mdp.modify_reward_std,
        # params={"term_name": "track_lin_vel_xy", "std": 0.25, "num_steps": 15000 * 24},
        # params={"term_name": "track_lin_vel_xy", "std": 0.25, "num_steps": 10000 * 24},
        params={"term_name": "track_lin_vel_xy", "std": 0.25, "num_steps": 20000 * 24},
    )

    track_ang_vel = CurrTerm(
        func=vel_mdp.modify_reward_std,
        # params={"term_name": "track_lin_vel_xy", "std": 0.25, "num_steps": 15000 * 24},
        # params={"term_name": "track_ang_vel_z", "std": 0.25, "num_steps": 10000 * 24},
        params={"term_name": "track_ang_vel_z", "std": 0.25, "num_steps": 20000 * 24},
    )

    track_heading = CurrTerm(
        func=vel_mdp.modify_reward_std,
        # params={"term_name": "track_heading", "std": 0.25, "num_steps": 15000 * 24},
        # params={"term_name": "track_heading", "std": 0.25, "num_steps": 10000 * 24},
        params={"term_name": "track_heading", "std": 0.25, "num_steps": 20000 * 24},
    )
