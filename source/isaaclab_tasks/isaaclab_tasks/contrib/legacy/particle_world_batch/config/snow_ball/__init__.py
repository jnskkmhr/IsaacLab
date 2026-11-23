# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for the `snow_ball` slope rolling task (`WorldParticle` and `Implicit MPM`)."""

import gymnasium as gym

from . import agents
from .snow_ball_env_cfg import SnowBallMPMEnvCfg, SnowBallParticleWorldEnvCfg

gym.register(
    id="IsaacContrib-ParticleWorld-SnowBall-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.snow_ball_env_cfg:SnowBallParticleWorldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SnowBallParticleWorldPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-ParticleWorld-SnowBall-MPM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.snow_ball_env_cfg:SnowBallMPMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SnowBallParticleWorldPPORunnerCfg",
    },
)

__all__ = ["SnowBallMPMEnvCfg", "SnowBallParticleWorldEnvCfg"]
