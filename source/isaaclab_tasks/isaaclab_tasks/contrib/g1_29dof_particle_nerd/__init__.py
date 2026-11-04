# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29-DoF locomotion over fixed granular terrain simulated with Particle NeRD."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

for task_id, cfg_name, env_class in (
    ("IsaacContrib-Velocity-Sand-Flat-G1-29dof-ParticleNeRD", "G1ParticleNeRDEnvCfg", "G1ParticleNeRDEnv"),
    ("IsaacContrib-Velocity-Sand-Ascending-G1-29dof-ParticleNeRD", "G1RoughParticleNeRDEnvCfg", "G1ParticleNeRDEnv"),
    ("IsaacContrib-Velocity-Sand-Descending-G1-29dof-ParticleNeRD", "G1DescendingParticleNeRDEnvCfg", "G1ParticleNeRDEnv"),
    ("IsaacContrib-Velocity-Sand-Dune-G1-29dof-ParticleNeRD", "G1DuneParticleNeRDEnvCfg", "G1ParticleNeRDEnv"),
    ("IsaacContrib-Balance-Sand-Tilting-G1-29dof-ParticleNeRD", "G1TiltingParticleNeRDEnvCfg", "G1TiltingParticleNeRDEnv"),
    ("IsaacContrib-Velocity-Sand-Tilting-G1-29dof-ParticleNeRD", "G1WalkingTiltingParticleNeRDEnvCfg", "G1TiltingParticleNeRDEnv"),
):
    for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"{task_id}{suffix}",
            entry_point=f"{__name__}.particle_nerd_env:{env_class}",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.particle_nerd_env_cfg:{cfg_name}{cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ParticleNeRDPPORunnerCfg",
            },
        )
