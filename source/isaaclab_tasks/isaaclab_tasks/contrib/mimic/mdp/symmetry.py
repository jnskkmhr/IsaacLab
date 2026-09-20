# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Sagittal reflection of motion-tracking body observations."""

from collections.abc import Sequence
from functools import lru_cache

import torch

from isaaclab_tasks.contrib.velocity.config.vel_mdp import mirror_vec3


def mirror_rotation_6d(data: torch.Tensor) -> torch.Tensor:
    """Reflect row-flattened first two rotation-matrix columns across the XZ plane.

    Args:
        data: Rotation representation, shape [..., 6].

    Returns:
        The first two columns of ``S @ R @ S``, where ``S = diag(1, -1, 1)``.
    """
    if data.shape[-1] != 6:
        raise ValueError(f"Expected six rotation components, got {data.shape}.")
    return data * data.new_tensor((1, -1, -1, 1, 1, -1))


def mirror_body_positions(data: torch.Tensor, body_names: Sequence[str]) -> torch.Tensor:
    """Swap paired bodies and reflect their anchor-frame positions across the XZ plane.

    Args:
        data: Flattened body positions [m], shape [..., 3 * len(body_names)].
        body_names: Body names in observation order, including both members of each pair.

    Returns:
        Reflected body positions [m] with the input shape.
    """
    bodies = _swap_bodies(data, body_names, 3)
    return mirror_vec3(bodies).reshape_as(data)


def mirror_body_orientations(data: torch.Tensor, body_names: Sequence[str]) -> torch.Tensor:
    """Swap paired bodies and reflect their anchor-frame 6D orientations.

    Args:
        data: Flattened body rotations, shape [..., 6 * len(body_names)].
        body_names: Body names in observation order, including both members of each pair.

    Returns:
        Reflected body rotations with the input shape.
    """
    bodies = _swap_bodies(data, body_names, 6)
    return mirror_rotation_6d(bodies).reshape_as(data)


def _swap_bodies(data: torch.Tensor, body_names: Sequence[str], width: int) -> torch.Tensor:
    if data.shape[-1] != width * len(body_names):
        raise ValueError(f"Expected {width * len(body_names)} body components, got {data.shape}.")
    permutation = _body_mirror(tuple(body_names))
    return data.reshape(*data.shape[:-1], len(body_names), width)[..., permutation, :]


@lru_cache(maxsize=16)
def _body_mirror(body_names: tuple[str, ...]) -> tuple[int, ...]:
    indices = {name: index for index, name in enumerate(body_names)}
    if len(indices) != len(body_names):
        raise ValueError("Body names must be unique.")
    permutation = []
    for name in body_names:
        if name.startswith("left_"):
            partner = "right_" + name.removeprefix("left_")
        elif name.startswith("right_"):
            partner = "left_" + name.removeprefix("right_")
        else:
            partner = name
        if partner not in indices:
            raise ValueError(f"Missing mirror body {partner!r} for {name!r}.")
        permutation.append(indices[partner])
    return tuple(permutation)
