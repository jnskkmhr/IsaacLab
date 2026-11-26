# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 towing a sled over rigid ground by a rope tied to its pelvis; the rigid counterpart of the sand task."""

import gymnasium as gym

for suffix, config in (("", "G1TowEnvCfg"), ("-Play", "G1TowEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Tow-G1-29dof-Rigid" + suffix,
        entry_point=f"{__name__}.tow_env:G1RigidTowEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tow_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1TowPPORunnerCfg",
        },
    )

# The walking task's rough terrain, a sled of random mass, and a quarter of the environments towing nothing.
gym.register(
    id="IsaacContrib-Tow-Rough-G1-29dof-Rigid",
    entry_point=f"{__name__}.tow_rough_env:G1RigidTowRoughEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tow_rough_env_cfg:G1TowRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1TowPPORunnerCfg",
    },
)
