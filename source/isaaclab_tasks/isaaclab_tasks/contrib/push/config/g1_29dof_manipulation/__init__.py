# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register G1 manipulation training and playback."""
import gymnasium as gym

for suffix, cfg in (("", "G1ManipulationEnvCfg"), ("-Play", "G1ManipulationEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Push-G1-29dof-Manipulation" + suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.manipulation_env_cfg:{cfg}",
            "comp_rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.compositional_rsl_rl_cfg:G1ManipulationCompositionalPPORunnerCfg"
            ),
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1ManipulationPPORunnerCfg",
        },
    )
