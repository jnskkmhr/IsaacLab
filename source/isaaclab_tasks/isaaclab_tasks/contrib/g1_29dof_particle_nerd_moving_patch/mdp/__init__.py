# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared locomotion MDP functions and the local terrain-aware reset term."""

__all__ = [
    "break_contact_penalty",
    "reward_soft_landing",
    "feet_air_time_positive_biped",
    "feet_pitch_contact",
    "foot_air_time",
    "foot_clearance_reward",
    "foot_contact",
    "foot_contact_force",
    "foot_contact_force_raw",
    "foot_touch_down_angle_penalty",
    "stance_foot_angle_penalty",
    "no_fly",
    "metric_sliderbar",
    "reset_root_state_on_terrain",
    "root_outside_workspace",
    "root_state_infinite",
    "terrain_levels_vel",
    "terrain_material_parameters",
]


from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_particle_cache.mdp.curriculums import terrain_levels_vel
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_particle_cache.mdp.observations import (
    foot_air_time,
    foot_contact,
    foot_contact_force,
    foot_contact_force_raw,
    terrain_material_parameters,
)
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_particle_cache.mdp.rewards import (
    break_contact_penalty,
    feet_air_time_positive_biped,
    feet_pitch_contact,
    foot_clearance_reward,
    foot_touch_down_angle_penalty,
    metric_sliderbar,
    no_fly,
    reward_soft_landing,
    stance_foot_angle_penalty,
)
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_particle_cache.mdp.terminations import (
    root_outside_workspace,
    root_state_infinite,
)

from .events import reset_root_state_on_terrain
