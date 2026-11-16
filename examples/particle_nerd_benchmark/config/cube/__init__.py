# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Gym registrations for the cube native MPM benchmarks."""

import gymnasium as gym

from particle_nerd_benchmark.core.case_config import make_config


def horizontal_flat_cfg():
    """Build the horizontal case on flat sand."""
    return make_config("horizontal", "flat")


def horizontal_ascend_cfg():
    """Build the horizontal case on ascend sand."""
    return make_config("horizontal", "ascend")


def horizontal_descend_cfg():
    """Build the horizontal case on descend sand."""
    return make_config("horizontal", "descend")


def horizontal_wave_cfg():
    """Build the horizontal case on wave sand."""
    return make_config("horizontal", "wave")


def vertical_flat_cfg():
    """Build the vertical case on flat sand."""
    return make_config("vertical", "flat")


def circle_flat_cfg():
    """Build the circle case on flat sand."""
    return make_config("circle", "flat")


def projectile_flat_cfg():
    """Build the projectile case on flat sand."""
    return make_config("projectile", "flat")


gym.register(
    id="IsaacBench-MPM-Box-Push-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:horizontal_flat_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "horizontal",
        "benchmark_terrain": "flat",
    },
)

gym.register(
    id="IsaacBench-MPM-Box-Push-Ascend-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:horizontal_ascend_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "horizontal",
        "benchmark_terrain": "ascend",
    },
)

gym.register(
    id="IsaacBench-MPM-Box-Push-Descend-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:horizontal_descend_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "horizontal",
        "benchmark_terrain": "descend",
    },
)

gym.register(
    id="IsaacBench-MPM-Box-Push-Wave-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:horizontal_wave_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "horizontal",
        "benchmark_terrain": "wave",
    },
)

gym.register(
    id="IsaacBench-MPM-Box-Intrude-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:vertical_flat_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "vertical",
        "benchmark_terrain": "flat",
    },
)

gym.register(
    id="IsaacBench-MPM-Box-Circle-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:circle_flat_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "circle",
        "benchmark_terrain": "flat",
    },
)

gym.register(
    id="IsaacBench-MPM-Box-Projectile-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:projectile_flat_cfg",
        "play_entry_point": "particle_nerd_benchmark.core.play",
        "benchmark_motion": "projectile",
        "benchmark_terrain": "flat",
    },
)
