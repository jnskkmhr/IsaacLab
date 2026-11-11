# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.commands.commands_cfg import UniformVelocityYawCommandCfg


@configclass
class G1CommandsCfg:
    base_velocity = UniformVelocityYawCommandCfg(
        debug_vis=True,
        asset_name="robot",
        heading_command=False,
        rel_heading_envs=1,
        rel_standing_envs=0.2,
        resampling_time_range=[10, 10],
        heading_control_stiffness=0.5,
        ranges=UniformVelocityYawCommandCfg.Ranges(
            heading=[-3.141592653589793, 3.141592653589793],
            ang_vel_z=[-1, 1],
            lin_vel_x=[-1, 1],
            lin_vel_y=[-0.5, 0.5],
        ),
    )
