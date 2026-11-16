# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the interactive box NeRD task."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-ParticleNeRD-Box-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.box_env_cfg:BoxParticleNeRDEnvCfg",
        "play_entry_point": "isaaclab_tasks.contrib.particle_nerd.config.play",
    },
)
