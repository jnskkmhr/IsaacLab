# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation configuration for the `box` two-way coupling task."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from ..mdp.observation import box_lin_vel, box_pose, box_target_error


@configclass
class BoxObservationsCfg:
    """Actor and critic observation groups for the `box` two-way coupling task."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        pose = ObservationTermCfg(func=box_pose)
        velocity = ObservationTermCfg(func=box_lin_vel)
        target_error = ObservationTermCfg(func=box_target_error)
        last_action = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False

    @configclass
    class CriticCfg(ObservationGroupCfg):
        pose = ObservationTermCfg(func=box_pose)
        velocity = ObservationTermCfg(func=box_lin_vel)
        target_error = ObservationTermCfg(func=box_target_error)
        last_action = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
