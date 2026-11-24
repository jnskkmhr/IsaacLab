# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms of the tow task: the walking task's terms on sand, plus the sled reset."""

from isaaclab_tasks.contrib.velocity_mpm.config.g1_29dof.mdp import (
    feet_air_time_positive_biped,
    feet_slide,
    foot_air_time,
    foot_clearance_reward,
    foot_contact,
    foot_contact_force,
    no_fly,
    reset_root_state_on_terrain,
    reward_soft_landing,
    root_outside_workspace,
    root_state_infinite,
    stance_foot_angle_penalty,
    terrain_material_parameters,
)

from .curriculums import sled_mass_levels
from .events import reset_object_behind_robot
from .rewards import lean_orientation_l2
