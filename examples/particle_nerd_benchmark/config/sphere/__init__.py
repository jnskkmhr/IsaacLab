# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Gym registrations for the sphere native MPM benchmarks."""

import gymnasium as gym

from particle_nerd_benchmark.core.case_config import make_config


def rolling_flat_cfg():
    """Build the rolling case on flat sand."""
    return make_config("rolling", "flat")


def rolling_ascend_cfg():
    """Build the rolling case on ascend sand."""
    return make_config("rolling", "ascend")


def rolling_descend_cfg():
    """Build the rolling case on descend sand."""
    return make_config("rolling", "descend")


def rolling_wave_cfg():
    """Build the rolling case on wave sand."""
    return make_config("rolling", "wave")


gym.register(
    id="IsaacBench-MPM-Sphere-Roll-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:rolling_flat_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "rolling",
        "benchmark_terrain": "flat",
    },
)

gym.register(
    id="IsaacBench-MPM-Sphere-Roll-Ascend-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:rolling_ascend_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "rolling",
        "benchmark_terrain": "ascend",
    },
)

gym.register(
    id="IsaacBench-MPM-Sphere-Roll-Descend-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:rolling_descend_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "rolling",
        "benchmark_terrain": "descend",
    },
)

gym.register(
    id="IsaacBench-MPM-Sphere-Roll-Wave-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:rolling_wave_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "rolling",
        "benchmark_terrain": "wave",
    },
)
