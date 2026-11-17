# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat moving-patch MPM box pushing with whole-body hand and velocity tracking."""

import gymnasium as gym

for suffix, config in (("", "G1ObjectPushEnvCfg"), ("-Play", "G1ObjectPushEnvCfg_PLAY")):
    gym.register(
        id=f"IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch{suffix}",
        entry_point=f"{__name__}.object_push_env:G1MovingPatchObjectPushEnv",
        disable_env_checker=True,
        kwargs={
            "default_agent": "rsl_rl",
            "env_cfg_entry_point": f"{__name__}.object_push_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_rsl_rl_cfg:G1ObjectPushPPORunnerCfg",
            "standard_rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1ObjectPushPPORunnerCfg",
            "comp_rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.compositional_rsl_rl_cfg:G1ObjectPushPPORunnerCfg"
            ),
        },
    )
