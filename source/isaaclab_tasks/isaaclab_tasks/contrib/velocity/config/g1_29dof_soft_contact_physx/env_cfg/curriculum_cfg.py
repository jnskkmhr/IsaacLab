# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as Term
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid import mdp as rigid_mdp


@configclass
class G1CurriculumCfg:
    terrain_levels = Term(func=rigid_mdp.terrain_levels_vel)

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

    track_lin_vel = Term(
        func=rigid_mdp.modify_reward_std, params={"std": 0.25, "num_steps": 480000, "term_name": "track_lin_vel_xy"}
    )

    track_ang_vel = Term(
        func=rigid_mdp.modify_reward_std, params={"std": 0.25, "num_steps": 480000, "term_name": "track_ang_vel_z"}
    )

    track_heading = Term(
        func=rigid_mdp.modify_reward_std, params={"std": 0.25, "num_steps": 480000, "term_name": "track_heading"}
    )
