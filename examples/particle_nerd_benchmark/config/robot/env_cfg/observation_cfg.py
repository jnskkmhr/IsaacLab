# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass


@configclass
class RobotObservationsCfg:
    @configclass
    class StateCfg(ObservationGroupCfg):
        joint_position = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_velocity = ObservationTermCfg(func=mdp.joint_vel_rel)
        enable_corruption = False

    state = StateCfg()
