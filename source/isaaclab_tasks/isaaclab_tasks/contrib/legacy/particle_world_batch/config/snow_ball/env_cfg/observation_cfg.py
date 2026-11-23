# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation configuration for the `snow_ball` slope rolling task."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from .. import mdp


@configclass
class SnowBallObservationsCfg:
    """Policy and critic observations for the rolling snow-ball on a cohesive snow slope."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        pose = ObservationTermCfg(func=mdp.snow_ball_pose)
        velocity = ObservationTermCfg(func=mdp.snow_ball_lin_vel)
        progress = ObservationTermCfg(func=mdp.snow_ball_slope_progress, params={"target_x": 0.65})
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()
