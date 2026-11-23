# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Granular foot-contact observations matching the reference MPM policy's critic."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


def _foot_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Read world-frame sand reaction forces [N], ordered like the articulation's ankle roll links."""
    if not hasattr(env, "_g1_sand_foot_feedback"):
        _, foot_names = env.scene["robot"].find_bodies(".*_ankle_roll_link")
        foot_columns = {name: column for column, name in enumerate(foot_names)}
        body_ids = torch.full((env.num_envs, len(foot_names)), -1, dtype=torch.long, device=env.device)
        pattern = re.compile(r"^/World/envs/env_(\d+)/Robot/(?:.*/)?([^/]+)$")
        for body_id, label in enumerate(NewtonManager.get_model().body_label):
            match = pattern.match(label)
            if match is not None and match[2] in foot_columns:
                body_ids[int(match[1]), foot_columns[match[2]]] = body_id
        if not foot_names or (body_ids < 0).any():
            raise RuntimeError("Could not resolve the G1 ankle roll links in the Newton model.")
        mapping = next(
            (
                mapping
                for mapping in NewtonManager._solver._proxy_mappings
                if mapping.src_name == "rigid" and mapping.dst_name == "sand"
            ),
            None,
        )
        if mapping is None:
            raise RuntimeError("G1 contact observations require the rigid-to-sand proxy mapping.")
        env._g1_sand_foot_feedback = (wp.to_torch(mapping.coupling_forces), body_ids)
    wrenches, body_ids = env._g1_sand_foot_feedback
    # Newton stores spatial wrenches as (force, torque). Reset worlds have no contact history.
    return torch.where(env.episode_length_buf[:, None, None] > 0, wrenches[body_ids, :3], 0.0)


def foot_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-foot sand contact flags using the reference policy's 5 N threshold."""
    return (torch.linalg.vector_norm(_foot_forces(env), dim=-1) > 5.0).float()


def foot_contact_force(env: ManagerBasedRLEnv, force_filter_threshold: float = 5.0) -> torch.Tensor:
    """Return signed log-compressed forces, clipping components at 1,000 N before filtering."""
    forces = _foot_forces(env).clamp(-1000.0, 1000.0).reshape(env.num_envs, -1)
    forces = forces * (forces.abs() > force_filter_threshold)
    return forces.sign() * torch.log1p(forces.abs())


class foot_air_time(ManagerTermBase):
    """Track time without sand contact [s], updating once per policy step and clearing on reset."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._air_time = torch.zeros_like(foot_contact(env))
        self._step = -1

    def reset(self, env_ids=None):
        self._air_time[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, filter_time: float = 0.5) -> torch.Tensor:
        if self._step != env.common_step_counter:
            airborne = (foot_contact(env) == 0) & (env.episode_length_buf[:, None] > 0)
            self._air_time = torch.where(airborne, self._air_time + env.step_dt, 0.0)
            self._step = env.common_step_counter
        return torch.where(self._air_time > filter_time, 0.0, self._air_time)
