# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recycling particle terrain with box-filtered rigid forces for hand-contact rewards."""

from isaaclab_tasks.contrib.push.config.g1_29dof_mpm_object_push.object_push_env import ObjectContactForces
from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.mpm_env import G1MovingPatchEnv

from .util.particles import ObjectPushMovingPatchParticles


class G1MovingPatchObjectPushEnv(ObjectContactForces, G1MovingPatchEnv):
    """Recycle moving-patch particles and distinguish hand-box contact from sand support."""

    moving_patch_particle_class = ObjectPushMovingPatchParticles
