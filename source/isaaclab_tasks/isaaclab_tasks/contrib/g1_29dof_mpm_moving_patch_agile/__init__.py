# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agile locomotion with moving-patch MJWarp–MPM coupling."""

import gymnasium as gym

for suffix, config in (("", "G1MovingPatchAgileEnvCfg"), ("-Play", "G1MovingPatchAgileEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-Agile" + suffix,
        entry_point=f"{__name__}.mpm_env:G1MovingPatchAgileEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.mpm_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_flash_sac_cfg:G1MovingPatchAgileFlashSACRunnerCfg",
            "rsl_rl_flash_sac_cfg_entry_point": (
                f"{__name__}.agents.compositional_flash_sac_cfg:G1MovingPatchAgileFlashSACRunnerCfg"
            ),
        },
    )


for suffix, config in (("", "G1MixedTerrainAgileEnvCfg"), ("-Play", "G1MixedTerrainAgileEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-Agile-MixedTerrain" + suffix,
        entry_point=f"{__name__}.mixed_env:G1MixedTerrainAgileEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_flash_sac_cfg:G1MovingPatchAgileFlashSACRunnerCfg",
            "rsl_rl_flash_sac_cfg_entry_point": f"{__name__}.agents.compositional_flash_sac_cfg:G1MovingPatchAgileFlashSACRunnerCfg",
        },
    )
