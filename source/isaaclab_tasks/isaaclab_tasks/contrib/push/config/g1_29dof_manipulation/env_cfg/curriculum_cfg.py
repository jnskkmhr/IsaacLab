# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tighten position and orientation tracking as learning progresses."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import ramp_reward_param


@configclass
class CurriculumCfg:
    left_position = CurrTerm(
        func=ramp_reward_param,
        params={
            "term_name": "left_position",
            "param_name": "std",
            "val_0": 0.5,
            "val_1": 0.05,
            "step_0": 0,
            "step_1": 120000,
        },
    )
    right_position = CurrTerm(
        func=ramp_reward_param,
        params={
            "term_name": "right_position",
            "param_name": "std",
            "val_0": 0.5,
            "val_1": 0.05,
            "step_0": 0,
            "step_1": 120000,
        },
    )
    left_orientation = CurrTerm(
        func=ramp_reward_param,
        params={
            "term_name": "left_orientation",
            "param_name": "std",
            "val_0": 1.0,
            "val_1": 0.35,
            "step_0": 0,
            "step_1": 120000,
        },
    )
    right_orientation = CurrTerm(
        func=ramp_reward_param,
        params={
            "term_name": "right_orientation",
            "param_name": "std",
            "val_0": 1.0,
            "val_1": 0.35,
            "step_0": 0,
            "step_1": 120000,
        },
    )
