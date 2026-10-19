# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 velocity tracking with CNN or attention-based height-map encoding."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Velocity-Rough-G1-29dof-Perception-AME",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughPerceptionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-G1-29dof-Perception-AME-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughPerceptionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-G1-29dof-Perception-CNN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughPerceptionCNNEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1CNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-G1-29dof-Perception-CNN-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughPerceptionCNNEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1CNNPPORunnerCfg",
    },
)
