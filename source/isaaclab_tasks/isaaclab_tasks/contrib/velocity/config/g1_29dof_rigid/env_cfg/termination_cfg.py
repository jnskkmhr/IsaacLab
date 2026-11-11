# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.vel_mdp import contact_terrain as contact_mdp
from isaaclab_tasks.contrib.velocity.config.vel_mdp.terminations import ExtremeJointPositionAction


@configclass
class G1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_too_low = DoneTerm(
        func=g1_mdp.root_height_below_minimum_adaptive,
        params={
            # "minimum_height": 0.5,
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 4})

    terrain_out_of_bounds = DoneTerm(func=mdp.terrain_out_of_bounds)


@configclass
class G1RoughTerminationsCfg:
    """Task-local rough-terrain configuration."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_too_low = DoneTerm(
        func=g1_mdp.root_height_below_minimum_adaptive,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg(name="robot", body_names=".*_ankle_roll_link")},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 4})
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"distance_buffer": 0.7, "asset_cfg": SceneEntityCfg(name="robot")},
    )
    extreme_action = DoneTerm(
        func=ExtremeJointPositionAction,
        params={"action_name": "joint_pos", "torque_fraction": 5.0, "asset_cfg": SceneEntityCfg(name="robot")},
    )
    solver_diverged = DoneTerm(func=contact_mdp.root_state_infinite, params={"asset_cfg": SceneEntityCfg(name="robot")})
