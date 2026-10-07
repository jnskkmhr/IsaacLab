# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Standalone term-wise sagittal reflection for WBC pose-goal PPO."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def mirror_vec3(data: torch.Tensor, axial: bool = False) -> torch.Tensor:
    """Reflect polar or axial vectors with trailing dimension three."""
    return data * data.new_tensor((-1, 1, -1) if axial else (1, -1, 1))


def mirror_rotation_6d(data: torch.Tensor) -> torch.Tensor:
    """Reflect row-flattened first two rotation columns across the sagittal plane."""
    return data * data.new_tensor((1, -1, -1, 1, 1, -1))


def mirror_g1_joints(data: torch.Tensor, joint_names: Sequence[str]) -> torch.Tensor:
    """Swap G1 left/right joints and negate roll/yaw coordinates."""
    partners = [
        name.replace("left_", "right_", 1) if name.startswith("left_") else name.replace("right_", "left_", 1)
        for name in joint_names
    ]
    permutation = [joint_names.index(name) for name in partners]
    signs = [-1 if name.endswith(("_roll_joint", "_yaw_joint")) else 1 for name in joint_names]
    return data[..., permutation] * data.new_tensor(signs)


def mirror_foot_scalars(data: torch.Tensor) -> torch.Tensor:
    """Swap left/right foot scalars."""
    return data.flip(-1)


def mirror_hand_positions(data: torch.Tensor) -> torch.Tensor:
    """Swap hands and reflect torso-relative XYZ positions."""
    return mirror_vec3(data.reshape(*data.shape[:-1], 2, 3).flip(-2)).reshape_as(data)


def mirror_hand_orientations(data: torch.Tensor) -> torch.Tensor:
    """Swap hands and reflect torso-relative 6D rotations."""
    return mirror_rotation_6d(data.reshape(*data.shape[:-1], 2, 6).flip(-2)).reshape_as(data)


def mirror_foot_forces(data: torch.Tensor) -> torch.Tensor:
    """Swap left/right feet and reflect their force vectors."""
    return mirror_vec3(data.reshape(*data.shape[:-1], 2, 3).flip(-2)).reshape_as(data)


def mirror_goal_command(data: torch.Tensor) -> torch.Tensor:
    """Reflect pelvis XYZ error, yaw sin/cos, torso rotation, and both hand goals."""
    return torch.cat(
        (
            mirror_vec3(data[..., :3]),
            data[..., 3:5] * data.new_tensor((-1, 1)),
            mirror_rotation_6d(data[..., 5:11]),
            mirror_hand_positions(data[..., 11:17]),
            mirror_hand_orientations(data[..., 17:29]),
        ),
        dim=-1,
    )


def compute_mirrored_states(
    env: ManagerBasedRLEnv, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append reflected observations/actions using this task's concatenated term layout."""
    env = env.unwrapped
    mirrored_obs = None
    if obs is not None:
        mirrored_obs = obs.clone()
        manager = env.observation_manager
        for group in obs.keys():
            if not manager.group_obs_concatenate[group]:
                raise ValueError("WBC symmetry requires concatenated observation groups")
            shapes = manager.group_obs_term_dim[group]
            if any(len(shape) != 1 for shape in shapes):
                raise ValueError("WBC symmetry requires flat observation terms")
            group_cfg = manager.cfg[group] if isinstance(manager.cfg, dict) else getattr(manager.cfg, group)
            chunks = torch.split(obs[group], [shape[0] for shape in shapes], dim=-1)
            values = []
            for name, value in zip(manager.active_terms[group], chunks):
                cfg = group_cfg[name] if isinstance(group_cfg, dict) else getattr(group_cfg, name)
                values.append(cfg.mirror(value, **cfg.mirror_params))
            mirrored_obs[group] = torch.cat(values, dim=-1)
        mirrored_obs = torch.cat((obs, mirrored_obs), dim=0)
    mirrored_actions = None
    if actions is not None:
        terms = env.action_manager.active_terms
        sizes = env.action_manager.action_term_dim
        chunks = torch.split(actions, sizes, dim=-1)
        values = []
        for name, value in zip(terms, chunks):
            cfg = env.action_manager.get_term(name).cfg
            values.append(cfg.mirror(value, **cfg.mirror_params))
        mirrored_actions = torch.cat((actions, torch.cat(values, dim=-1)), dim=0)
    return mirrored_obs, mirrored_actions


@configclass
class MirrorObservationTermCfg(ObservationTermCfg):
    """Flat WBC observation term with its sagittal reflection rule."""

    mirror: Callable[..., torch.Tensor] = MISSING
    mirror_params: dict[str, Any] = {}


@configclass
class MirrorJointPositionActionCfg(JointPositionActionCfg):
    """Joint position actions with explicit G1 mirror metadata."""

    mirror: Callable[..., torch.Tensor] = MISSING
    mirror_params: dict[str, Any] = {}
