# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic rigid and coupled moving-patch experiments."""

import gymnasium as gym

for terrain, env_cfg, runner_cfg, entry_point in (
    ("Rigid", "G1SuperMimicEnvCfg", "G1SuperMimicRunnerCfg", "isaaclab.envs:ManagerBasedRLEnv"),
    (
        "MixedTerrain",
        "G1MixedSuperMimicEnvCfg",
        "G1MixedSuperMimicRunnerCfg",
        "isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_agile.mixed_env:G1MixedTerrainAgileEnv",
    ),
):
    gym.register(
        id=f"IsaacContrib-Velocity-G1-29dof-SuperMimic-{terrain}",
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_supermimic_cfg:{runner_cfg}",
            "rsl_rl_supermimic_cfg_entry_point": f"{__name__}.agents.compositional_supermimic_cfg:{runner_cfg}",
        },
    )
