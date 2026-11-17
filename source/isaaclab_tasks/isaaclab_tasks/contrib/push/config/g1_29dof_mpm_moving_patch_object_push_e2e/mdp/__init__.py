# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sand-surface box reset and moving-patch boundary terminations; pushing terms come from ``g1_29dof_object_push_e2e``."""

from isaaclab_tasks.contrib.push.config.g1_29dof_mpm_moving_patch_object_push.mdp.terminations import (
    bodies_outside_patch,
    object_outside_patch,
    root_outside_workspace,
)
from isaaclab_tasks.contrib.push.mdp.terminations import box_tilted

from .events import (
    AssistivePushForce,
    BoxMassCurriculum,
    assistive_push_force_state,
    box_mass_curriculum_state,
    reset_box_on_sand,
)
from .terminations import robot_velocity_implausible
