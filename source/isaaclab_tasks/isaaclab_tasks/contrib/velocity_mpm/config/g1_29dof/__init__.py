# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified G1 MPM locomotion task; the MPM solver settings choose how the sand is simulated."""

import gymnasium as gym

# Streaming MPM
for suffix, config in (("", "G1MPMEnvCfg"), ("-Play", "G1MPMEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Velocity-G1-29dof-MPM" + suffix,
        entry_point=f"{__name__}.streaming_mpm_env:G1MPMEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.streaming_mpm_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1MovingPatchPPORunnerCfg",
            "rsl_rl_standard_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1MovingPatchPPORunnerCfg",
        },
    )

# Streaming MPM on mixed terrain: every rough terrain type once as rigid ground and once as sand.
for suffix, config in (("", "G1MPMMixedEnvCfg"), ("-Play", "G1MPMMixedEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Velocity-G1-29dof-MPM-Mixed" + suffix,
        entry_point=f"{__name__}.streaming_mpm_mixed_env:G1MPMMixedEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.streaming_mpm_mixed_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1MixedTerrainPPORunnerCfg",
            "comp_rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1MixedTerrainPPORunnerCfg",
        },
    )

# Fixed MPM: every spawned particle is simulated and optionally freeze settled particles.
for name, config, env_class in (
    ("Flat", "G1TestbedEnvCfg", "G1TestbedEnv"),
    ("Uphill", "G1UphillTestbedEnvCfg", "G1TestbedEnv"),
    ("Downhill", "G1DownhillTestbedEnvCfg", "G1TestbedEnv"),
    ("Dune", "G1DuneTestbedEnvCfg", "G1TestbedEnv"),
    ("Tilting-Balance", "G1TiltingTestbedEnvCfg", "G1TiltingTestbedEnv"),
    ("Tilting", "G1WalkingTiltingTestbedEnvCfg", "G1TiltingTestbedEnv"),
):
    for suffix, config_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"IsaacContrib-Velocity-G1-29dof-MPM-{name}{suffix}",
            entry_point=f"{__name__}.mpm_env:{env_class}",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.mpm_env_cfg:{config}{config_suffix}",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1TestbedPPORunnerCfg",
            },
        )
