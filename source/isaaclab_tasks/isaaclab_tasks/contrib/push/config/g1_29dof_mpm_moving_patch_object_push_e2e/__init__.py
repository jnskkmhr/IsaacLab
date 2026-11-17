# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-policy PPO box pushing on recycling moving-patch MPM sand."""

import gymnasium as gym

for suffix, config in (
    ("", "G1ObjectPushEnvCfg"),
    ("-Assist", "G1ObjectPushAssistEnvCfg"),
    ("-Play", "G1ObjectPushEnvCfg_PLAY"),
):
    gym.register(
        id=f"IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch-E2E{suffix}",
        entry_point=f"{__name__}.object_push_env:G1MovingPatchObjectPushE2EEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.object_push_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1ObjectPushPPORunnerCfg",
        },
    )
