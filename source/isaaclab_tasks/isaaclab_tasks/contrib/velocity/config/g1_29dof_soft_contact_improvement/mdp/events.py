# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton foot-collider margins selected by the assigned flat-terrain column."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from newton import ModelFlags

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class set_terrain_foot_margin(ManagerTermBase):
    """Set foot margins once; the depth curriculum preserves each environment's terrain type.

    This changes Newton rigid contacts with the hard floor. The analytical RFT
    surface and its intrusion-depth calculation are unchanged.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        self._manager = env.sim.physics_manager
        self._margin = wp.to_torch(asset._root_view.get_attribute("shape_margin", self._manager.get_model())[:, 0])
        body_shapes = asset._root_view.body_shapes
        body_ids = asset.map_body_ids_to_backend(asset_cfg.body_ids)
        self._shape_ids = torch.tensor(
            [shape for body in body_ids for shape in body_shapes[body]], device=env.device, dtype=torch.long
        )
        if self._shape_ids.numel() == 0:
            raise ValueError("The selected feet have no Newton shapes.")
        terrain = env.scene[cfg.params["terrain_name"]]
        generator = terrain.cfg.terrain_generator
        if generator is None or not generator.curriculum:
            raise ValueError("Terrain-conditioned margins require curriculum terrain columns.")
        names = list(generator.sub_terrains)
        soft_names = cfg.params["soft_terrain_names"]
        if not set(soft_names).issubset(names):
            raise ValueError(f"Unknown soft terrain names: {soft_names}")
        proportions = np.array([sub.proportion for sub in generator.sub_terrains.values()])
        # Match TerrainGenerator._generate_curriculum_terrains' column assignment.
        sub_indices = np.searchsorted(
            np.cumsum(proportions / proportions.sum()),
            np.arange(generator.num_cols) / generator.num_cols + 0.001,
            side="right",
        )
        self._soft_columns = torch.tensor(
            [names[index] in soft_names for index in sub_indices], device=env.device, dtype=torch.bool
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        terrain_name: str,
        soft_terrain_names: tuple[str, ...],
        rigid_margin: float,
        soft_margin: float,
    ) -> None:
        """Write margins [m] in place and notify the rigid solver of changed shape properties."""
        if rigid_margin < 0.0 or soft_margin < 0.0:
            raise ValueError("Foot contact margins must be nonnegative.")
        ids = torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
        soft = self._soft_columns[env.scene[terrain_name].terrain_types[ids]]
        margins = torch.where(soft, soft_margin, rigid_margin)
        self._margin[ids[:, None], self._shape_ids[None, :]] = margins[:, None]
        self._manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
