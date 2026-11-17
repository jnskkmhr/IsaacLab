# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CommandTermCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp

from ..mdp.commands import UniformHeightCommand


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Desired root height above the ray-hit rigid surface [m]."""

    class_type: type = UniformHeightCommand
    asset_name: str = "robot"
    sensor_name: str = "height_scanner"
    height_range: tuple[float, float] = (0.60, 0.80)


@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = g1_mdp.UniformVelocityYawCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=g1_mdp.UniformVelocityYawCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )

    base_height = UniformHeightCommandCfg(resampling_time_range=(10.0, 10.0), debug_vis=False)
