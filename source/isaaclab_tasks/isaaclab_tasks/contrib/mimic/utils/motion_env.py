# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mimic environment entry point with terrain selection after configuration overrides."""

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from .motion_paired_terrain_cfg import configure_motion_paired_terrain
from .motion_paired_terrain_generator import motion_files_use_terrain


class MotionTrackingEnv(ManagerBasedRLEnv):
    """Build terrain from the final motion command configuration before spawning robots."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        motion = cfg.commands.motion
        if motion_files_use_terrain(motion.motion_files, motion.terrain_mode):
            configure_motion_paired_terrain(cfg, motion.motion_files)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
