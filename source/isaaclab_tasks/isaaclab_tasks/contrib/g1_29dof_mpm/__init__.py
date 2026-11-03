# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29-DoF locomotion over granular media simulated with a coupled MJWarp/MPM solver."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

for task_id, cfg_name, env_class in (
    ("IsaacContrib-Velocity-Sand-Flat-G1-29dof-MPM", "G1MPMEnvCfg", "G1MPMEnv"),
    ("IsaacContrib-Velocity-Sand-Ascending-G1-29dof-MPM", "G1RoughMPMEnvCfg", "G1MPMEnv"),
    ("IsaacContrib-Velocity-Sand-Descending-G1-29dof-MPM", "G1DescendingMPMEnvCfg", "G1MPMEnv"),
    ("IsaacContrib-Velocity-Sand-Dune-G1-29dof-MPM", "G1DuneMPMEnvCfg", "G1MPMEnv"),
    ("IsaacContrib-Balance-Sand-Tilting-G1-29dof-MPM", "G1TiltingMPMEnvCfg", "G1TiltingMPMEnv"),
    ("IsaacContrib-Velocity-Sand-Tilting-G1-29dof-MPM", "G1WalkingTiltingMPMEnvCfg", "G1TiltingMPMEnv"),
):
    for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"{task_id}{suffix}",
            entry_point=f"{__name__}.g1_mpm_env:{env_class}",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.g1_mpm_env_cfg:{cfg_name}{cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.compositional_ppo_cfg:G1MPMPPORunnerCfg",
            },
        )
