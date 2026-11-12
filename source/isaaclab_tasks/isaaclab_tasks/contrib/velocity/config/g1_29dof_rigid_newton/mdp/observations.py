# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation layout and material normalization from reference run 4blqnxbi."""

import torch


def foot_air_time(env, sensor_cfg):
    """Unfiltered current foot air time [s], matching the reference rigid branch."""
    return env.scene[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]


def foot_contact_forces(env, sensor_cfg, force_filter_threshold=5.0):
    """Reference positive-component filtering and signed log compression."""
    forces = env.scene[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids].clamp(-1000.0, 1000.0)
    forces = torch.where(forces > force_filter_threshold, forces, 0.0)
    return (torch.sign(forces) * torch.log1p(forces.abs())).reshape(env.num_envs, -1)


def foot_contact_forces_raw(env, sensor_cfg, force_filter_threshold=5.0):
    """Unfiltered world contact forces for logging [N]."""
    return env.scene[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids].reshape(env.num_envs, -1)


def terrain_material_parameters(env, sensor_cfg):
    """Rigid ground is one in the reference normalized-stiffness observation."""
    return torch.ones((env.num_envs, 1), device=env.device)
