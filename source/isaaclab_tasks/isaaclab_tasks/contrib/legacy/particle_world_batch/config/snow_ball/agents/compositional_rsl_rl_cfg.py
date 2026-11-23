# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compositional RSL-RL PPO configuration for the `snow_ball` slope rolling task."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import SnowBallParticleWorldPPORunnerCfg


@configclass
class SnowBallParticleWorldCompositionalPPORunnerCfg(SnowBallParticleWorldPPORunnerCfg):
    experiment_name = "particle_world_snow_ball_compositional"
