# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Disturbances for soft-contact motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp.observations import foot_contact_hybrid

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .actions import AnklePitchPerturbation


def push_by_setting_velocity_in_stance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    sensor_cfg: SceneEntityCfg,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply interval pushes only during fully weighted stance with at least one grounded foot.

    Contact is selected from the active rigid or soft solver. Events outside stance or without
    foot contact are skipped, not deferred to touchdown. The stance blend is excluded so pushes
    do not interfere with the transition to motion tracking.

    Args:
        env: Environment to perturb.
        env_ids: Environments whose push interval has elapsed.
        velocity_range: Per-axis velocity increment ranges [m/s] for x/y/z and [rad/s] for
            roll/pitch/yaw. Omitted axes receive no increment.
        sensor_cfg: Rigid contact sensor with foot bodies in soft-solver foot order.
        command_name: Motion command defining the stance intervals.
        asset_cfg: Robot receiving the velocity increment.
    """
    command = env.command_manager.get_term(command_name)
    env_ids = env_ids[command.standing_weight[env_ids] >= 1.0 - 1.0e-6]
    if env_ids.numel() == 0:
        return
    contacts = foot_contact_hybrid(env, rigid_contact_sensor_cfg=sensor_cfg)
    env_ids = env_ids[contacts[env_ids].bool().any(dim=-1)]
    if env_ids.numel() == 0:
        return
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)


def push_body_in_stance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]] | tuple[float, float],
    torque_range: tuple[float, float],
    sensor_cfg: SceneEntityCfg,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Push selected bodies for one physics step during fully weighted, grounded stance.

    Forces and torques are sampled independently for each axis, body, and environment
    and added to the instantaneous wrench buffer in the world frame. Forces act at
    each body's center of mass. Events outside stance or without foot contact are
    skipped, not deferred to touchdown.

    Args:
        env: Environment to perturb.
        env_ids: Environments whose push interval has elapsed.
        force_range: Per-axis world-frame force ranges [N] keyed by ``x``, ``y``,
            and ``z``. Omitted axes receive zero force. A tuple applies the same
            range to all three axes.
        torque_range: Uniform torque component range [N m].
        sensor_cfg: Rigid contact sensor with foot bodies in soft-solver foot order.
        command_name: Motion command defining stance intervals.
        asset_cfg: Asset and bodies receiving the push. Body names are resolved to
            ``asset_cfg.body_ids`` by the event manager; the default selects all bodies.
    """
    command = env.command_manager.get_term(command_name)
    env_ids = env_ids[command.standing_weight[env_ids] >= 1.0 - 1.0e-6]
    if env_ids.numel() == 0:
        return
    contacts = foot_contact_hybrid(env, rigid_contact_sensor_cfg=sensor_cfg)
    env_ids = env_ids[contacts[env_ids].bool().any(dim=-1)]
    if env_ids.numel() == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_bodies = len(range(asset.num_bodies)[body_ids]) if isinstance(body_ids, slice) else len(body_ids)
    if num_bodies == 0:
        return
    size = (len(env_ids), num_bodies, 3)
    if isinstance(force_range, dict):
        ranges = torch.tensor([force_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")], device=asset.device)
        forces = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], size, asset.device)
    else:
        forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    asset.instantaneous_wrench_composer.add_forces_and_torques_index(
        forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids, is_global=True
    )


def perturb_ankle_pitch_in_stance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    torque_range: tuple[float, float],
    duration_range_s: tuple[float, float],
    action_name: str = "ankle_pitch_perturbation",
) -> None:
    """Trigger smooth external ankle-pitch torque pulses during grounded stance.

    Requires an :class:`AnklePitchPerturbation` action after the soft-contact action.
    It applies local-y torques each physics step, independently of contact wrenches.
    Active pulses are not restarted, and unsupported feet are not perturbed.

    Args:
        env: Environment to perturb.
        env_ids: Environments whose perturbation interval has elapsed.
        torque_range: Uniform signed peak torque range per foot [N m].
        duration_range_s: Uniform pulse duration range [s], at least one physics step.
        action_name: Action term maintaining and applying the pulse state.
    """
    action: AnklePitchPerturbation = env.action_manager.get_term(action_name)
    action.trigger(env_ids, torque_range, duration_range_s)
