# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for the `box` two-way coupling task."""

from .compositional_rsl_rl_cfg import BoxParticleWorldCompositionalPPORunnerCfg
from .rsl_rl_ppo_cfg import BoxParticleWorldPPORunnerCfg

__all__ = ["BoxParticleWorldCompositionalPPORunnerCfg", "BoxParticleWorldPPORunnerCfg"]
