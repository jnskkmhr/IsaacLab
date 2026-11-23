# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configurations for the `box` two-way coupling task."""

from .compositional_rsl_rl_cfg import BoxParticleNeRDCompositionalPPORunnerCfg
from .rsl_rl_ppo_cfg import BoxParticleNeRDPPORunnerCfg

__all__ = [
    "BoxParticleNeRDCompositionalPPORunnerCfg",
    "BoxParticleNeRDPPORunnerCfg",
]
