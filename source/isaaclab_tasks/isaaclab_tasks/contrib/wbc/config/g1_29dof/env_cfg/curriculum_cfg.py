# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.wbc.mdp as mdp


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the G1 whole-body task."""

    target_joint_change = CurriculumTermCfg(
        func=mdp.target_joint_change,
        params={"final_max_joint_rms": 1.0, "num_steps": 1000 * 24},
    )
