# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from .. import mdp as goal_mdp


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    task_resets = CurrTerm(func=goal_mdp.task_reset_difficulty, params={"command_name": "motion"})
