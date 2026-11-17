# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

for suffix, config in (("", "G1FlatEnvCfg"), ("-Play", "G1FlatEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Improvement-Tyiby5k6" + suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
    )
