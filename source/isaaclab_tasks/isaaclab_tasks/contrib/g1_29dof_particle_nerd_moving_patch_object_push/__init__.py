# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direction-conditioned pushing on a recycling Particle NeRD bed."""

import gymnasium as gym

for suffix, config in (("", "ObjectPushEnvCfg"), ("-Play", "ObjectPushEnvCfg_PLAY")):
    gym.register(
        id=f"IsaacContrib-Push-G1-29dof-ParticleNeRD-MovingPatch-ObjectPush{suffix}",
        entry_point=f"{__name__}.object_push_env:ObjectPushEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.object_push_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.compositional_rsl_rl_cfg:G1ObjectPushPPORunnerCfg",
        },
    )
