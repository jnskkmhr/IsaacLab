# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Adapted from Agile. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp as g1_mdp


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=g1_mdp.terrain_levels_vel_curriculum,
        params={
            "command_name": "base_velocity",
            "move_up_distance": 4.0,
            "move_down_distance": 2.0,
            "n_successes": 1,
            "n_failures": 10,
            "p_random_move_up": 0.00,
            "p_random_move_down": 0.00,
        },
    )

    increase_action_rate_regularization = CurrTerm(
        func=g1_mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate",
            "start_step": 2_500,
            "num_steps": 5_000,
            "terminal_weight": -1.0,
            "use_log_space": False,
        },
    )

    increase_action_rate_rate_regularization = CurrTerm(
        func=g1_mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate_rate",
            "start_step": 3_000,
            "num_steps": 5_000,
            "terminal_weight": -0.5,
            "use_log_space": False,
        },
    )
