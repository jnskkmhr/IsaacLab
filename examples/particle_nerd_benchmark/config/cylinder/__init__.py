# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Registered cylinder interaction with the same native MPM flat sand bed."""

import gymnasium as gym

from particle_nerd_benchmark.core.case_config import make_config


def horizontal_flat_cfg():
    """Build a vertically oriented cylinder pushed horizontally through sand."""
    return make_config("horizontal", "flat", geometry="cylinder")


gym.register(
    id="IsaacBench-MPM-Cylinder-Push-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:horizontal_flat_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "horizontal",
        "benchmark_terrain": "flat",
        "benchmark_geometry": "cylinder",
    },
)
