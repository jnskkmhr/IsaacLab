# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 towing a sled over streaming MPM sand by a rope tied to its pelvis."""

import gymnasium as gym

for suffix, config in (("", "G1TowEnvCfg"), ("-Play", "G1TowEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Tow-G1-29dof-MPM" + suffix,
        entry_point=f"{__name__}.tow_env:G1TowEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tow_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1TowPPORunnerCfg",
            "rsl_rl_standard_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1TowPPORunnerCfg",
        },
    )

# Mixed terrain: half of the environments tow on rigid ground, half on sand.
for suffix, config in (("", "G1TowMixedEnvCfg"), ("-Play", "G1TowMixedEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Tow-G1-29dof-MPM-Mixed" + suffix,
        entry_point=f"{__name__}.tow_mixed_env:G1TowMixedEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tow_mixed_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1TowPPORunnerCfg",
            "rsl_rl_standard_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1TowPPORunnerCfg",
        },
    )

# Mixed terrain with a towed object that differs between environments: a box, a sphere or a wheel.
for suffix, config in (("", "G1TowObjectsEnvCfg"), ("-Play", "G1TowObjectsEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Tow-G1-29dof-MPM-Mixed-Objects" + suffix,
        entry_point=f"{__name__}.tow_objects_env:G1TowObjectsEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tow_objects_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1TowPPORunnerCfg",
            "rsl_rl_standard_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1TowPPORunnerCfg",
        },
    )
