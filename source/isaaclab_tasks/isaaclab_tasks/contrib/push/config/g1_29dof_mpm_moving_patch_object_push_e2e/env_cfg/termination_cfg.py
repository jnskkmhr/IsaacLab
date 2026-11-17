# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.contrib.velocity.config.vel_mdp as vel_mdp
import isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.mdp as mpm_mdp
import isaaclab_tasks.core.velocity.mdp as mdp

from .. import mdp as push_mdp
from .scene_cfg import BOX_SIZE

MAX_PLAUSIBLE_JOINT_VEL = 100.0
"""Joint speed above which the simulation is treated as blown up [rad/s]; about three times the fastest G1 joint."""

MAX_PLAUSIBLE_ROOT_LIN_VEL = 10.0
"""Base linear speed above which the simulation is treated as blown up [m/s]."""

MAX_PLAUSIBLE_ROOT_ANG_VEL = 30.0
"""Base angular speed above which the simulation is treated as blown up [rad/s]."""


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
    terrain_out_of_bounds = TerminationTermCfg(
        func=push_mdp.root_outside_workspace,
        params={"margin": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )

    extreme_action = TerminationTermCfg(
        func=vel_mdp.ExtremeJointPositionAction,  # type: ignore
        params={
            "action_name": "joint_pos",
            "torque_fraction": 5.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # a diverged granular solve leaves the world NaN, where every thresholded term stays false
    solver_diverged = TerminationTermCfg(
        func=mpm_mdp.root_state_infinite,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # a blow-up can also leave the state finite but orders of magnitude beyond the hardware
    implausible_velocity = TerminationTermCfg(
        func=push_mdp.robot_velocity_implausible,
        params={
            "max_joint_vel": MAX_PLAUSIBLE_JOINT_VEL,
            "max_root_lin_vel": MAX_PLAUSIBLE_ROOT_LIN_VEL,
            "max_root_ang_vel": MAX_PLAUSIBLE_ROOT_ANG_VEL,
        },
    )

    box_tilted = TerminationTermCfg(func=push_mdp.box_tilted, params={"limit": 0.5})
    box_outside_patch = TerminationTermCfg(
        func=push_mdp.object_outside_patch,
        params={"object_name": "box", "half_extents": tuple(size / 2 for size in BOX_SIZE), "margin": 0.05},
    )
    feet_outside_patch = TerminationTermCfg(
        func=push_mdp.bodies_outside_patch,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "margin": 0.2,
        },
    )
    # End the episode before the object or feet lose fully simulated sand support.
    box_outside_workspace = TerminationTermCfg(
        func=push_mdp.root_outside_workspace,
        params={"margin": 0.5, "asset_cfg": SceneEntityCfg("box")},
    )
    box_state_infinite = TerminationTermCfg(
        func=mpm_mdp.root_state_infinite,
        params={"asset_cfg": SceneEntityCfg("box")},
    )
