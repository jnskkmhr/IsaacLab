# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp


@configclass
class G1CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    command_vel = None
    increase_action_rate_regularization = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={"term_name": "action_rate", "weight_0": -0.25, "weight_1": -1.0, "step_0": 2500, "step_1": 7500},
    )
    increase_action_rate_rate_regularization = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={"term_name": "action_rate_rate", "weight_0": -0.025, "weight_1": -0.5, "step_0": 3000, "step_1": 8000},
    )
