# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import paired motion terrain through the standard scene terrain interface."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporter

from .motion_paired_terrain_generator import MotionPairedTerrainGenerator

if TYPE_CHECKING:
    from .motion_paired_terrain_cfg import MotionPairedTerrainImporterCfg


class MotionPairedTerrainImporter(TerrainImporter):
    """Share static terrain across environments and assign origins by motion ID."""

    def __init__(self, cfg: MotionPairedTerrainImporterCfg):
        # Geometry comes from reference files, not TerrainImporter's procedural generator.
        cfg.validate()
        if cfg.collision_group != -1 or cfg.disable_collider:
            raise ValueError("Paired terrain requires a shared collider (collision_group=-1).")
        self.cfg = cfg
        self.device = SimulationContext.instance().device
        self.terrain_prim_paths = []
        self._terrain_flat_patches = {}
        layout = MotionPairedTerrainGenerator(cfg.motion_files, cfg.patch_margin, cfg.patch_spacing)
        self.motion_files = layout.motion_files
        self.motion_paired_terrain_ids = torch.as_tensor(layout.motion_paired_terrain_ids, device=self.device)
        self.motion_origins = torch.as_tensor(layout.motion_origins, device=self.device)
        self.terrain_origins = torch.as_tensor(layout.terrain_origins[:, None, :], device=self.device)
        self.env_origins = self.motion_origins[0].repeat(cfg.num_envs, 1)
        self.import_mesh("terrain", layout.terrain_mesh)
        self.set_debug_vis(cfg.debug_vis)

    def validate_motion_files(self, motion_files: Sequence[str]) -> None:
        """Reject motion/terrain lists with different files or different ordering."""
        paths = tuple(str(Path(path).expanduser().resolve()) for path in motion_files)
        if paths != self.motion_files:
            raise ValueError("Motion command and terrain importer must use the same ordered motion_files list.")

    def assign_motion_origins(self, env_ids: torch.Tensor, motion_ids: torch.Tensor) -> None:
        """Select patch translations [m] before writing the selected robots' reset states."""
        self.env_origins[env_ids] = self.motion_origins[motion_ids]

    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        """Reject terrain curricula that would break the motion-to-terrain pairing."""
        raise RuntimeError("Paired terrain origins are selected by motion ID, not terrain difficulty.")
