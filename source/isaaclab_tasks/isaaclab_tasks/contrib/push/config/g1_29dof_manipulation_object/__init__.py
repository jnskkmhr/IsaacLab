# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the frozen-policy upper-body pushing evaluation."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Push-G1-29dof-Manipulation-Object-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manipulation_object_env_cfg:G1ManipulationObjectEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1ManipulationPPORunnerCfg",
    },
)
