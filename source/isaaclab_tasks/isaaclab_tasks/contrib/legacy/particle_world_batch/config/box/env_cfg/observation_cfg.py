# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation configuration for the `box` two-way coupling task."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from .. import mdp


@configclass
class BoxObservationsCfg:
    """Policy and critic observations for the rigid box on sand."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        pose = ObservationTermCfg(func=mdp.box_pose)
        velocity = ObservationTermCfg(func=mdp.box_lin_vel)
        target_error = ObservationTermCfg(func=mdp.box_target_error, params={"target_pos": (0.0, 0.0, 0.08)})
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()
