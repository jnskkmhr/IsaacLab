# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 tasks using Newton GPU body-following particle regions."""

import gymnasium as gym

from . import agents

for terrain, cfg_name in (
    ("Flat", "G1MovingPatchEnvCfg"),
    ("Ascending", "G1RoughMPMEnvCfg"),
    ("Descending", "G1DescendingMovingPatchEnvCfg"),
    ("Dune", "G1DuneMovingPatchEnvCfg"),
):
    for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"IsaacContrib-Velocity-Sand-{terrain}-G1-29dof-MPM-MovingPatch{suffix}",
            entry_point=f"{__name__}.mpm_env:G1MovingPatchEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.mpm_env_cfg:{cfg_name}{cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.compositional_ppo_cfg:G1MPMPPORunnerCfg",
            },
        )


for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"IsaacContrib-Velocity-Generator-G1-29dof-MPM{suffix}",
        entry_point=f"{__name__}.mpm_env:G1MovingPatchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.mpm_env_cfg:G1GeneratedMPMEnvCfg{cfg_suffix}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.compositional_ppo_cfg:G1MPMPPORunnerCfg",
        },
    )
