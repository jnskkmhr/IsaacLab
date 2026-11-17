# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm.mdp.observations import (
    foot_contact, foot_contact_force, foot_contact_force_raw, foot_air_time, terrain_material_parameters,
)
from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm.mdp.terminations import (
    root_state_not_finite,
    root_outside_workspace,
)
from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.mdp.rewards import feet_slide
from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch_particle_cache.mdp.rewards import (
    feet_air_time_positive_biped, no_fly, reward_soft_landing, stance_foot_angle_penalty,
)
from .commands import UniformHeightCommand
from .rewards import energy, action_rate_l2, commanded_base_height_exp, foot_clearance_reward
