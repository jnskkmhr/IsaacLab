# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.curriculum_cfg import G1CurriculumCfg as RigidCurriculumCfg
from isaaclab_tasks.contrib.mimic.mdp.curriculums import terrain_levels_motion_success


@configclass
class G1CurriculumCfg(RigidCurriculumCfg):
    """Increase soft-layer depth when the full reference is completed without failure."""

    terrain_levels = CurriculumTermCfg(
        func=terrain_levels_motion_success,
        params={"command_name": "motion"},
    )
