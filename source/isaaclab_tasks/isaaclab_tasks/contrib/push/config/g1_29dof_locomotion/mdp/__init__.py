# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 locomotion mdp."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from isaaclab_tasks.contrib.push.mdp.rewards import (
    body_ang_vel_xy_l2,
    body_flat_orientation_l2,
    body_lin_vel_z_l2,
)

from .rewards import *  # noqa: F401, F403
