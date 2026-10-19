# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 SE(2) + Z goal reaching with CNN or AME terrain perception."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-AME",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughPerceptionReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-AME-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ame_play_env_cfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1AMEPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-CNN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:cnn_env_cfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1CNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-CNN-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:cnn_play_env_cfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_ppo_cfg:G1CNNPPORunnerCfg",
    },
)
