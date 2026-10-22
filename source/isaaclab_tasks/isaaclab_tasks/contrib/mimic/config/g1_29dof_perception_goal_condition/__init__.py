# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition",
    entry_point="isaaclab_tasks.contrib.mimic.utils.motion_env:MotionTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition-Play",
    entry_point="isaaclab_tasks.contrib.mimic.utils.motion_env:MotionTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PPORunnerCfg",
    },
)
