# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation configuration for the `go2` quadruped granular locomotion task."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from ..mdp.observation import go2_base_pos


@configclass
class Go2ObservationsCfg:
    """Actor and critic observation groups for the `go2` quadruped granular locomotion task."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_pos = ObservationTermCfg(func=go2_base_pos)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False

    @configclass
    class CriticCfg(ObservationGroupCfg):
        base_pos = ObservationTermCfg(func=go2_base_pos)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
