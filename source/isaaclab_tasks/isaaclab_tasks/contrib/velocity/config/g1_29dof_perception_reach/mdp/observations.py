# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Elevation observations relative to the torso-link origin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class elevation_map(ManagerTermBase):
    """XYZ grid in meters relative to the scanner origin, with yaw-only axes.

    For this task the scanner origin is the torso-link origin. Grid axis 0 is
    forward X and axis 1 is left Y; Z is world-up height relative to that origin.
    Missed rays keep their grid XY and use ``missing_height`` for Z. The sensor's
    ray starts may be elevated above its origin without changing this reference.
    Return ``(B, nx, ny, 3)`` for AME or height-only ``(B, 1, nx, ny)`` when
    ``height_only`` is enabled for RSL-RL's CNN.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        pattern = self._sensor.cfg.pattern_cfg
        if (
            self._sensor.cfg.ray_alignment != "yaw"
            or not isinstance(pattern, patterns.GridPatternCfg)
            or pattern.ordering != "yx"
            or tuple(pattern.direction) != (0.0, 0.0, -1.0)
        ):
            raise ValueError("Elevation maps require a yaw-aligned, downward grid with ordering='yx'.")
        starts, _ = pattern.func(pattern, env.device)
        self._shape = (starts[:, 0].unique().numel(), starts[:, 1].unique().numel(), 3)
        if starts.shape[0] != self._sensor.num_rays:
            raise ValueError("Elevation-map grid must match the sensor ray count.")
        self._missing = starts.clone()
        self._missing[:, 2] = cfg.params["missing_height"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        missing_height: float,
        height_only: bool = False,
    ) -> torch.Tensor:
        data = self._sensor.data
        hits = data.ray_hits_w.torch
        valid = torch.isfinite(hits).all(dim=-1)
        relative = torch.where(valid.unsqueeze(-1), hits - data.pos_w.torch[:, None, :], 0.0)
        quat = yaw_quat(data.quat_w.torch)[:, None, :].expand(-1, hits.shape[1], -1)
        points = quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape_as(hits)
        points = torch.where(valid.unsqueeze(-1), points, self._missing.unsqueeze(0))
        points = points.reshape(-1, *self._shape)
        return points[..., 2].unsqueeze(1).contiguous() if height_only else points
