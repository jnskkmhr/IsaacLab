# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for the `newton_box` task: Newton's `mpm_twoway_coupling` scene with Particle NeRD Local or MPM."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-ParticleNeRDLocal-NewtonBox-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.newton_box_env_cfg:NewtonBoxEnvCfg"},
)

gym.register(
    id="IsaacContrib-ParticleNeRDLocal-NewtonBox-MPM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.newton_box_env_cfg:NewtonBoxMPMEnvCfg"},
)
