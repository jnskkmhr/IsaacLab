# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for the batched `franka` sand-pushing task with ParticleNeRD-Local."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-ParticleNeRDLocal-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_env_cfg:FrankaParticleNeRDLocalEnvCfg",
        "play_entry_point": "isaaclab_tasks.contrib.particle_nerd_local.config.play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaParticleNeRDLocalPPORunnerCfg",
        "compositional_rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.compositional_rsl_rl_cfg:FrankaParticleNeRDLocalCompositionalPPORunnerCfg"
        ),
    },
)

gym.register(
    id="IsaacContrib-ParticleNeRDLocal-Franka-MPM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_env_cfg:FrankaMPMCouplerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaParticleNeRDLocalPPORunnerCfg",
        "compositional_rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.compositional_rsl_rl_cfg:FrankaParticleNeRDLocalCompositionalPPORunnerCfg"
        ),
    },
)
