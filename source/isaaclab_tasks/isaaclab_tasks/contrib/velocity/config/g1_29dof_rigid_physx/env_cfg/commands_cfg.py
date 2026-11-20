# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .. import mdp as vel_mdp


@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = vel_mdp.UniformVelocityYawCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=vel_mdp.UniformVelocityYawCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )
    foot_height = vel_mdp.SwingCommandCfg(foot_height=(0.1, 0.1), resampling_time_range=(10000000000.0, 10000000000.0))


import math


@configclass
class G1FlatCommandsCfg(G1CommandsCfg):
    base_velocity = G1CommandsCfg().base_velocity.replace(
        ranges=G1CommandsCfg().base_velocity.ranges.replace(
            lin_vel_x=(-1.0, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        )
    )


@configclass
class G1FlatPlayCommandsCfg(G1FlatCommandsCfg):
    base_velocity = G1FlatCommandsCfg().base_velocity.replace(
        ranges=G1FlatCommandsCfg().base_velocity.ranges.replace(
            lin_vel_x=(2.0, 2.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        heading_command=False,
        rel_standing_envs=0.0,
        resampling_time_range=(2.5, 2.5),
    )


@configclass
class G1RoughPlayCommandsCfg(G1CommandsCfg):
    base_velocity = G1CommandsCfg().base_velocity.replace(
        ranges=G1CommandsCfg().base_velocity.ranges.replace(
            lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(0.0, 0.0)
        )
    )
