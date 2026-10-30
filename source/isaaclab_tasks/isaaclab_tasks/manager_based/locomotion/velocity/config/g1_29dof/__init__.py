# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

"""
vanilla policy
"""

gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_recurrent_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerRecurrentCfg",
        "rsl_rl_adaptation_cfg_entry_point": f"{agents.__name__}.rsl_rl_adaptation_cfg:G1AdaptationPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg"
        ),
        "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_fastsac_cfg:G1HolosomaFastSACAgentCfg",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_ppo_cfg:G1HolosomaPPOAgentCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_recurrent_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerRecurrentCfg",
        "rsl_rl_adaptation_cfg_entry_point": f"{agents.__name__}.rsl_rl_adaptation_cfg:G1AdaptationPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg"
        ),
        "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_fastsac_cfg:G1HolosomaFastSACAgentCfg",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_ppo_cfg:G1HolosomaPPOAgentCfg",
    },
)

"""
distillation policy
"""


gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Teacher-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_distillation:G1FlatTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_recurrent_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerRecurrentCfg",
        "rsl_rl_adaptation_cfg_entry_point": f"{agents.__name__}.rsl_rl_adaptation_cfg:G1AdaptationPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg"
        ),
        "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_fastsac_cfg:G1HolosomaFastSACAgentCfg",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_ppo_cfg:G1HolosomaPPOAgentCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Teacher-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_distillation:G1FlatTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_recurrent_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerRecurrentCfg",
        "rsl_rl_adaptation_cfg_entry_point": f"{agents.__name__}.rsl_rl_adaptation_cfg:G1AdaptationPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg"
        ),
        "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_fastsac_cfg:G1HolosomaFastSACAgentCfg",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_ppo_cfg:G1HolosomaPPOAgentCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Student-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_distillation:G1FlatStudentEnvCfg",
        "rsl_rl_adaptation_cfg_entry_point": f"{agents.__name__}.rsl_rl_adaptation_cfg:G1AdaptationDistillationRunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Student-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_distillation:G1FlatStudentEnvCfg_PLAY",
        "rsl_rl_adaptation_cfg_entry_point": f"{agents.__name__}.rsl_rl_adaptation_cfg:G1AdaptationDistillationRunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:G1FlatDistillationRunnerCfg"
        ),
    },
)
