# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the initial flat-sand pushing stage."""

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch.mdp as mpm_mdp
import isaaclab_tasks.core.velocity.mdp as core_mdp

from .. import mdp


@configclass
class EventsCfg:
    """Reset the robot before placing the box ahead of it."""

    physics_material = None
    add_base_mass = None
    reset_base = EventTermCfg(
        func=mpm_mdp.reset_root_state_on_terrain,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTermCfg(
        func=core_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    scale_actuator_gains = None
    mpm_material = None
    mpm_material_log = None
    push_robot = None
    reset_box = EventTermCfg(func=mdp.reset_box, mode="reset")
