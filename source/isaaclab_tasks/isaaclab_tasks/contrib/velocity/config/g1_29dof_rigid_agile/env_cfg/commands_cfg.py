# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp as g1_mdp


@configclass
class G1CommandsCfg:
    base_velocity = g1_mdp.UniformNullVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.25,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=g1_mdp.UniformNullVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )
