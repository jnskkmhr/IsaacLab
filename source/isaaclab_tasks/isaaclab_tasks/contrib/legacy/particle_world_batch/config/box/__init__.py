# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for the `box` two-way coupling task (`WorldParticle` and `Implicit MPM`)."""

import gymnasium as gym

from . import agents
from .box_env_cfg import BoxMPMEnvCfg, BoxParticleWorldEnvCfg

gym.register(
    id="IsaacContrib-ParticleWorld-Box-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.box_env_cfg:BoxParticleWorldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoxParticleWorldPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-ParticleWorld-Box-MPM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.box_env_cfg:BoxMPMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoxParticleWorldPPORunnerCfg",
    },
)

__all__ = ["BoxMPMEnvCfg", "BoxParticleWorldEnvCfg"]
