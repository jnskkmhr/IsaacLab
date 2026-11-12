# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Newton",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"},
)


gym.register(
    id="IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Newton-Symmetry",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_symmetry_env_cfg:G1FlatSymmetryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1FlatSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-G1-29dof-SoftContact-Newton",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
