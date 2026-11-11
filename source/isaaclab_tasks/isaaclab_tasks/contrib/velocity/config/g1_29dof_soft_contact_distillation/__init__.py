# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat rigid/RFT teacher and student tasks; the unqualified IDs default to PhysX."""

import gymnasium as gym

for backend, prefix in (("", "G1"), ("-Newton", "G1Newton")):
    for stage in ("Teacher", "Student"):
        gym.register(
            id=f"IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-{stage}{backend}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.distillation_env_cfg:{prefix}{stage}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_rsl_rl_cfg:G1{stage}RunnerCfg",
            },
        )
