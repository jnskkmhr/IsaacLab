# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Gym registrations for the robot native MPM benchmarks."""

import gymnasium as gym

from particle_nerd_benchmark.core.case_config import make_config


def franka_flat_cfg():
    """Build the franka case on flat sand."""
    return make_config("franka", "flat")


def quadruped_flat_cfg():
    """Build the quadruped case on flat sand."""
    return make_config("quadruped", "flat")


def g1_flat_cfg():
    """Build the g1 case on flat sand."""
    return make_config("g1", "flat")


gym.register(
    id="IsaacBench-MPM-Franka-Push-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:franka_flat_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "franka",
        "benchmark_terrain": "flat",
    },
)

gym.register(
    id="IsaacBench-MPM-Go2-Crawl-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:quadruped_flat_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "quadruped",
        "benchmark_terrain": "flat",
    },
)

gym.register(
    id="IsaacBench-MPM-G1-Walk-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:g1_flat_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "g1",
        "benchmark_terrain": "flat",
    },
)
