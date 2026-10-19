# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_reach.mdp as reach_mdp


@configclass
class G1CommandsCfg:
    """One SE(2) + Z base-pose target per episode."""

    base_pose = reach_mdp.TerrainPoseCommandCfg()
