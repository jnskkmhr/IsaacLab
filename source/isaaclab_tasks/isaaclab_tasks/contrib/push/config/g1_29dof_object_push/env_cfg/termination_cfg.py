# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.contrib.velocity.config.vel_mdp as vel_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.push import mdp as push_mdp

from .action_cfg import ALL_JOINTS


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_too_low = TerminationTermCfg(
        func=g1_mdp.root_height_below_minimum_adaptive,
        params={
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.pi / 4})

    extreme_action = TerminationTermCfg(
        func=vel_mdp.ExtremeJointPositionAction,  # type: ignore
        params={
            "action_name": "joint_pos",
            "torque_fraction": 5.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True),
        },
    )

    # Explicitly catch nonfinite robot or box state.
    solver_diverged = TerminationTermCfg(func=push_mdp.invalid_root_state)

    box_tilted = TerminationTermCfg(func=push_mdp.box_tilted)
