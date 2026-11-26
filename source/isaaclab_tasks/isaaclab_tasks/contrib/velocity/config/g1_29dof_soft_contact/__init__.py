# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

for terrain in ("Flat", "Rough"):
    for suffix, config_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"IsaacContrib-Velocity-{terrain}-G1-29dof-SoftContact{suffix}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{terrain.lower()}_env_cfg:G1{terrain}EnvCfg{config_suffix}",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1{terrain}PPORunnerCfg",
            },
        )
