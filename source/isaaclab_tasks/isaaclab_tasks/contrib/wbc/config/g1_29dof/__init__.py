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
