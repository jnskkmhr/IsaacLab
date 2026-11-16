# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configurations for the `g1` humanoid granular locomotion task."""

from .compositional_rsl_rl_cfg import G1ParticleNeRDLocalCompositionalPPORunnerCfg
from .rsl_rl_ppo_cfg import G1ParticleNeRDLocalPPORunnerCfg

__all__ = [
    "G1ParticleNeRDLocalCompositionalPPORunnerCfg",
    "G1ParticleNeRDLocalPPORunnerCfg",
]
