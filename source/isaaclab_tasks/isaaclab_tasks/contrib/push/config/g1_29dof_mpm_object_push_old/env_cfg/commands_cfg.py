# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
from isaaclab_tasks.contrib.push.mdp.commands import BoxHandPoseCommandCfg


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = g1_mdp.UniformVelocityYawCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=g1_mdp.UniformVelocityYawCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.4),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )

    hand_pose = BoxHandPoseCommandCfg()
