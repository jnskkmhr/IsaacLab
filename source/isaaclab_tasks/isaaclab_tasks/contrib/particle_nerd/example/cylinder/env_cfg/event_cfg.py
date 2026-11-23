# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event configuration for the `cylinder` two-way coupling task."""

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..mdp.observation import reset_sand_particles


@configclass
class CylinderEventsCfg:
    """Reset the rigid cylinder state and granular sand bed on episode reset."""

    reset_cylinder = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.10, 0.10), "z": (0.0, 0.05)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("cylinder"),
        },
    )
    reset_sand = EventTermCfg(func=reset_sand_particles, mode="reset")
