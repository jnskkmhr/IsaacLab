# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot and box resets, and per-environment box mass."""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp as push_mdp


@configclass
class EventsCfg:
    """Sample the box mass once per environment; reset the robot before placing the box."""

    box_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "operation": "abs",
            "mass_distribution_params": (5.0, 15.0),
        },
    )
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    # Box center ahead of the pelvis: the rear face starts 0.10-0.40 m beyond the wrists.
    reset_box = EventTermCfg(
        func=push_mdp.reset_box,
        mode="reset",
        params={
            "distance_range": (0.95, 1.25),
            "lateral_range": (-0.10, 0.10),
            "yaw_range": (-math.radians(15.0), math.radians(15.0)),
        },
    )
