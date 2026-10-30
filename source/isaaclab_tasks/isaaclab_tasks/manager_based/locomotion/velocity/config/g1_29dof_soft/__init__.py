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
    id="Isaac-Velocity-Flat-G1-29dof-Soft-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_fastsac_cfg:G1HolosomaFastSACAgentCfg",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_ppo_cfg:G1HolosomaPPOAgentCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Soft-Finetune-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_FINETUNE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfgFinetune",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_fastsac_cfg:G1HolosomaFastSACAgentCfg",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_ppo_cfg:G1HolosomaPPOAgentCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Soft-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_fastsac_cfg:G1HolosomaFastSACAgentCfg",
        # "holosoma_agent_cfg_entry_point": f"{agents.__name__}.holosoma_agent_ppo_cfg:G1HolosomaPPOAgentCfg",
    },
)

# gym.register(
#     id="Isaac-Velocity-Flat-G1-29dof-Soft-Symmetry-v1",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerWithSymmetryCfg",
#     },
# )


# gym.register(
#     id="Isaac-Velocity-Flat-G1-29dof-Soft-Symmetry-Play-v1",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerWithSymmetryCfg",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Flat-G1-29dof-Soft-Recurrent-v1",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatRecurrentPPOEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_recurrent_ppo_cfg:G1FlatPPORunnerRecurrentCfg",
#     },
# )


# gym.register(
#     id="Isaac-Velocity-Flat-G1-29dof-Soft-Recurrent-Play-v1",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatRecurrentPPOEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_recurrent_ppo_cfg:G1FlatPPORunnerRecurrentCfg",
#     },
# )


"""
distillation policy
"""

# teacher
gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Soft-Teacher-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_distillation:G1FlatTeacherEnvCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Soft-Teacher-Finetune-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_distillation:G1FlatTeacherEnvCfg_FINETUNE",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1PPORunnerCfgFinetune",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Soft-Teacher-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_distillation:G1FlatTeacherEnvCfg_PLAY",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1PPORunnerCfg",
    },
)

# student
gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Soft-Student-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_distillation:G1FlatStudentEnvCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1DistillationRunnerCfg",
        "rsl_rl_ppo_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1PPODistillationRunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-29dof-Soft-Student-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_distillation:G1FlatEnvStudentCfg_PLAY",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1DistillationRunnerCfg",
        "rsl_rl_ppo_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:G1PPODistillationRunnerCfg",
    },
)