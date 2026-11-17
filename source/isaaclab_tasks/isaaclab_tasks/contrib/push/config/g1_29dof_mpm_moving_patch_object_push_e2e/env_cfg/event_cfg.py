# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot and box resets on the sand surface; particle states are recycled by the environment."""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch import mdp as particle_mdp

from .. import mdp as push_mdp
from .commands_cfg import STANDING_COMMAND_THRESHOLD

BOX_MASS_START_RANGE = (2.0, 5.0)
"""Box mass range at the start of the curriculum [kg]; a 5 kg box slides on sand under about 30 N."""

BOX_MASS_FINAL_RANGE = (5.0, 15.0)
"""Box mass range at the end of the curriculum [kg]; the rigid-ground range."""

BOX_MASS_PROMOTE_RATIO = 0.6
"""Episode box speed ratio above which the mass range widens."""

BOX_MASS_DEMOTE_RATIO = 0.3
"""Episode box speed ratio below which the mass range narrows."""

BOX_MASS_LEVEL_STEP = 0.002
"""Curriculum progress per finished episode; 500 net successful episodes per process reach the final range."""

ASSIST_ETA = 0.7
"""Box speed ratio at which the assistive push force is fully retired."""

ASSIST_BETA_MAX = 0.8
"""Upper bound of the assistive gain; below 1 so the robot always supplies part of the push."""

ASSIST_VELOCITY_GAIN = 10.0
"""Assistive force per unit robot mass and unit base velocity error [1/s]."""

ASSIST_MAX_FORCE = 150.0
"""Bound on the assistive force before the gain is applied [N]; under half the robot weight."""


@configclass
class EventsCfg:
    """Resample the box mass from its curriculum range; reset the robot on the sand before placing the box."""

    # Listed first: the mass is in place before the box is put on the sand.
    box_mass = EventTerm(
        func=push_mdp.BoxMassCurriculum,
        mode="reset",
        params={
            "command_name": "base_velocity",
            "start_range": BOX_MASS_START_RANGE,
            "final_range": BOX_MASS_FINAL_RANGE,
            "promote_ratio": BOX_MASS_PROMOTE_RATIO,
            "demote_ratio": BOX_MASS_DEMOTE_RATIO,
            "level_step": BOX_MASS_LEVEL_STEP,
        },
    )

    reset_base = EventTerm(
        func=particle_mdp.reset_root_state_on_terrain,
        mode="reset",
        params={"pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)}, "velocity_range": {}},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    # Box center ahead of the pelvis: the rear face starts 0.10-0.40 m beyond the wrists.
    reset_box = EventTerm(
        func=push_mdp.reset_box_on_sand,
        mode="reset",
        params={
            "distance_range": (0.95, 1.25),
            "lateral_range": (-0.10, 0.10),
            "yaw_range": (-math.radians(15.0), math.radians(15.0)),
        },
    )


@configclass
class AssistEventsCfg(EventsCfg):
    """Sand resets plus the training-only assistive push force."""

    assistive_push_force = EventTerm(
        func=push_mdp.AssistivePushForce,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "command_name": "base_velocity",
            "eta": ASSIST_ETA,
            "beta_max": ASSIST_BETA_MAX,
            "k_d": ASSIST_VELOCITY_GAIN,
            "max_force": ASSIST_MAX_FORCE,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
