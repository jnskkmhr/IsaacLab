# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation configuration for the `particle` task."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from ..mdp.observation import block_center_of_mass


@configclass
class ParticleObservationsCfg:
    """Single observation group: centre of mass of the sand block."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        center_of_mass = ObservationTermCfg(func=block_center_of_mass)
        enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
