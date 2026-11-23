# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration for the `particle` task."""

from isaaclab.utils import configclass


@configclass
class ParticleRewardsCfg:
    """No reward terms; the scene is for simulation and evaluation only."""
