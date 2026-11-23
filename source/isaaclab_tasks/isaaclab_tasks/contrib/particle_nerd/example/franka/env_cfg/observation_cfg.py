# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation configuration for the `franka` sand-pushing task."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from ..mdp.observation import franka_ee_pos


@configclass
class FrankaObservationsCfg:
    """Actor and critic observation groups for the `franka` sand-pushing task."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        ee_pos = ObservationTermCfg(func=franka_ee_pos)
        last_action = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False

    @configclass
    class CriticCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        ee_pos = ObservationTermCfg(func=franka_ee_pos)
        last_action = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
