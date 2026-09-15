# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""G1 reflection rules used by the generic term-driven augmentation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from isaaclab.sensors.ray_caster.patterns import GridPatternCfg, grid_pattern

from isaaclab_tasks.contrib.velocity.config.vel_mdp import (
    compute_mirrored_states,
    mirror_joints,
    mirror_quat,
    mirror_vec3,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "compute_symmetric_states",
    "mirror_g1_joints",
    "mirror_velocity_heading",
    "mirror_foot_scalars",
    "mirror_foot_forces",
    "mirror_height_scan",
]


def compute_symmetric_states(
    env: ManagerBasedRLEnv, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append reflected samples using the mirror rules on observation and action terms."""
    return compute_mirrored_states(env, obs, actions)


def mirror_g1_joints(data: torch.Tensor, joint_names: Sequence[str]) -> torch.Tensor:
    """Reflect G1 joint values in the explicitly configured joint order.

    Roll and yaw axes change sign; pitch, knee, and elbow axes keep their signs.
    Both members of each left/right pair must be present. Values retain their units.
    """
    permutation, signs = _joint_mirror(tuple(joint_names))
    return mirror_joints(data, permutation, signs)


def mirror_velocity_heading(data: torch.Tensor) -> torch.Tensor:
    """Reflect ``[vx, vy, wz, qx, qy, qz, qw]`` commands, including the heading."""
    if data.shape[-1] != 7:
        raise ValueError("Expected three velocity commands followed by an XYZW heading.")
    velocity = data[..., :3] * data.new_tensor((1, -1, -1))
    return torch.cat((velocity, mirror_quat(data[..., 3:])), dim=-1)


def mirror_foot_scalars(data: torch.Tensor) -> torch.Tensor:
    """Swap the two foot scalar channels, preserving their units."""
    return mirror_joints(data, (1, 0))


def mirror_foot_forces(data: torch.Tensor) -> torch.Tensor:
    """Swap feet and reflect flattened force vectors (including signed-log forces)."""
    if data.shape[-1] != 6:
        raise ValueError("Expected two three-component foot forces.")
    vectors = data.reshape(*data.shape[:-1], 2, 3).flip(-2)
    return mirror_vec3(vectors).reshape_as(data)


def mirror_height_scan(data: torch.Tensor, pattern_cfg: GridPatternCfg) -> torch.Tensor:
    """Reflect flattened height samples using the configured grid size and ordering."""
    permutation = _scan_mirror(tuple(pattern_cfg.size), pattern_cfg.resolution, pattern_cfg.ordering)
    return mirror_joints(data, permutation)


@lru_cache(maxsize=32)
def _joint_mirror(joint_names: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    indices = {name: i for i, name in enumerate(joint_names)}
    if len(indices) != len(joint_names):
        raise ValueError("Joint names must be unique.")
    permutation, signs = [], []
    for name in joint_names:
        if name.startswith("left_"):
            partner = "right_" + name.removeprefix("left_")
        elif name.startswith("right_"):
            partner = "left_" + name.removeprefix("right_")
        else:
            partner = name
        if partner not in indices:
            raise ValueError(f"Missing mirror joint {partner!r} for {name!r}.")
        if name.endswith(("_roll_joint", "_yaw_joint")):
            sign = -1
        elif name.endswith(("_pitch_joint", "_knee_joint", "_elbow_joint")):
            sign = 1
        else:
            raise ValueError(f"No G1 mirror axis rule for joint {name!r}.")
        permutation.append(indices[partner])
        signs.append(sign)
    return tuple(permutation), tuple(signs)


@lru_cache(maxsize=16)
def _scan_mirror(size: tuple[float, float], resolution: float, ordering: str) -> tuple[int, ...]:
    cfg = GridPatternCfg(size=size, resolution=resolution, ordering=ordering) # type: ignore
    positions, _ = grid_pattern(cfg, "cpu")
    nx, ny = (positions[:, axis].unique().numel() for axis in (0, 1))
    shape, axis = ((ny, nx), 0) if ordering == "xy" else ((nx, ny), 1)
    return tuple(torch.arange(len(positions)).reshape(shape).flip(axis).flatten().tolist())
