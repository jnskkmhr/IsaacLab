# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configurations for the `go2` quadruped granular locomotion task."""

from .compositional_rsl_rl_cfg import Go2ParticleNeRDCompositionalPPORunnerCfg
from .rsl_rl_ppo_cfg import Go2ParticleNeRDPPORunnerCfg

__all__ = [
    "Go2ParticleNeRDCompositionalPPORunnerCfg",
    "Go2ParticleNeRDPPORunnerCfg",
]
