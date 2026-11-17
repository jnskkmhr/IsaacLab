# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register whole-body box pushing with multiple frozen teachers."""

import gymnasium as gym

for suffix, cfg in (("", "G1ObjectPushEnvCfg"), ("-Play", "G1ObjectPushEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Push-G1-29dof-ObjectPush" + suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.object_push_env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_rsl_rl_cfg:G1ObjectPushPPORunnerCfg",
            "comp_rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_rsl_rl_cfg:G1ObjectPushPPORunnerCfg",
        },
    )
