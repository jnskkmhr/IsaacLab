# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reuse backend-independent reference observation layouts and reward functions."""

from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact_physx.mdp import (
    action_rate_l2,
    reward_feet_pitch_contact_hybrid,
    root_quat_wxyz,
    terrain_material_parameters_hybrid,
    velocity_commands_wxyz,
)
