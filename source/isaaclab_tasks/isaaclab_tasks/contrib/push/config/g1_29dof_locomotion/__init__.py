# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register G1 locomotion training and playback."""
import gymnasium as gym

for suffix, cfg in (("", "G1LocomotionEnvCfg"), ("-Play", "G1LocomotionEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Push-G1-29dof-Locomotion" + suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:{cfg}",
            "comp_rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.compositional_rsl_rl_cfg:G1LocomotionCompositionalPPORunnerCfg"
            ),
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1LocomotionPPORunnerCfg",
            # "rsl_rl_sac_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:G1LocomotionSACRunnerCfg",
        },
    )
