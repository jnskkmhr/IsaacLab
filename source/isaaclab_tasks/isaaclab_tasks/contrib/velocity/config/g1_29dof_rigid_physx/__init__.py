# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference rigid-terrain G1 tasks using Isaac Sim PhysX."""
import gymnasium as gym

for terrain in ("Flat", "Rough"):
    for play in (False, True):
        gym.register(
            id=f"IsaacContrib-Velocity-{terrain}-G1-29dof-Rigid-PhysX" + ("-Play" if play else ""),
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{terrain.lower()}_env_cfg:G1{terrain}EnvCfg" + ("_PLAY" if play else ""),
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1{terrain}PPORunnerCfg",
            },
        )
