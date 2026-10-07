# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="IsaacContrib-WBC-G1-29dof",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:G1WholeBodyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:G1WholeBodyTeacherRunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:G1WholeBodyStudentRunnerCfg",
    },
)


gym.register(
    id="IsaacContrib-WBC-G1-29dof-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:G1WholeBodyEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:G1WholeBodyTeacherRunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:G1WholeBodyStudentRunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-WBC-PoseGoal-G1-29dof",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_goal_env_cfg:G1PoseGoalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.pose_goal_rl_cfg:G1PoseGoalRunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-WBC-PoseGoal-G1-29dof-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_goal_env_cfg:G1PoseGoalEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.pose_goal_rl_cfg:G1PoseGoalRunnerCfg",
    },
)
