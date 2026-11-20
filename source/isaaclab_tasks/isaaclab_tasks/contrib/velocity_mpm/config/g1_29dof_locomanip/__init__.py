# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug task: the streaming G1 MPM task with free objects, to exercise several bodies on the sand."""

import gymnasium as gym

from .. import g1_29dof

for suffix, config in (("", "G1LocomanipEnvCfg"), ("-Play", "G1LocomanipEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Velocity-G1-29dof-MPM-Locomanip" + suffix,
        entry_point=f"{g1_29dof.__name__}.streaming_mpm_env:G1MPMEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.streaming_mpm_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{g1_29dof.__name__}.agents.compositional_ppo_cfg:G1MovingPatchPPORunnerCfg",
            "rsl_rl_standard_cfg_entry_point": f"{g1_29dof.__name__}.agents.rsl_rl_ppo_cfg:G1MovingPatchPPORunnerCfg",
        },
    )
