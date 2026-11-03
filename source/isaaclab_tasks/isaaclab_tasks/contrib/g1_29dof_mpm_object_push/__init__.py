# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat moving-patch MPM box pushing with whole-body hand and velocity tracking."""

import gymnasium as gym

for suffix, config in (("", "G1ObjectPushEnvCfg"), ("-Play", "G1ObjectPushEnvCfg_PLAY")):
    gym.register(
        id=f"IsaacContrib-Push-Box-G1-29dof-MPM{suffix}",
        entry_point="isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mpm_env:G1MovingPatchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ObjectPushPPORunnerCfg",
        },
    )
