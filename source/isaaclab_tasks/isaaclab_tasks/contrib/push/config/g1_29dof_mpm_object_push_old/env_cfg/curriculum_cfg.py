# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curricula disabled for initial box-contact learning."""

from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    """Keep commands, rewards, and unbiased gravity fixed during this stage."""

    terrain_levels = None
    command_vel = None
    gravity_roll_bias = None
    gravity_pitch_bias = None
    track_lin_vel = None
    track_ang_vel = None
    track_heading = None
    track_lin_vel_weight = None
