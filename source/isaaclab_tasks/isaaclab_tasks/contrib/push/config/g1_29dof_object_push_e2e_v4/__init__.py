# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 whole-body box pushing on rigid flat ground with a tighter foot-trailing band."""

import gymnasium as gym

for suffix, config in (("", "G1ObjectPushEnvCfg"), ("-Play", "G1ObjectPushEnvCfg_PLAY")):
    gym.register(
        id=f"IsaacContrib-Push-Box-Flat-G1-29dof-V4{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.object_push_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1ObjectPushPPORunnerCfg",
            "comp_rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_rsl_rl_cfg:G1ObjectPushPPORunnerCfg",
        },
    )
