# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_reach.mdp as reach_mdp


@configclass
class G1CurriculumCfg:
    """Change terrain difficulty using final-state goal pose accuracy."""

    terrain_levels = CurrTerm(func=reach_mdp.terrain_levels_reach, params={"command_name": "base_pose"})
