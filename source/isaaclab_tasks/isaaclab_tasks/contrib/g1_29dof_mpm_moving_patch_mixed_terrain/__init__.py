# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""One G1 policy trained on rigid contact and deformable MPM terrain."""

import gymnasium as gym

for suffix, config in (("", "G1MixedTerrainEnvCfg"), ("-Play", "G1MixedTerrainEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain" + suffix,
        entry_point=f"{__name__}.mixed_env:G1MixedTerrainEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.mixed_env_cfg:G1MixedTerrainPPORunnerCfg",
        },
    )
