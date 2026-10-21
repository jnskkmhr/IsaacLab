# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in configuration for loading terrain paired with motion clips."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg


@configclass
class MotionPairedTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for static terrain assembled from the reference NPZ files."""

    class_type: str = "{DIR}.motion_paired_terrain_importer:MotionPairedTerrainImporter"
    motion_files: list[str] = []
    """NPZ paths, in exactly the same order as the motion command's files."""
    patch_margin: float = 1.0
    """Extra distance around the motion and terrain bounds [m]."""
    patch_spacing: float = 2.0
    """Distance between adjacent patch bounds [m]."""


def configure_motion_paired_terrain(
    env_cfg: ManagerBasedRLEnvCfg, motion_files: Sequence[str], command_name: str = "motion"
) -> None:
    """Pair a mimic command with shared terrain before constructing the environment.

    Call after applying command-line overrides. Reference XY/yaw offsets are
    disabled in matching reset events; other robot perturbations remain intact.
    Both motion and geometry receive the same translation at episode reset.
    """
    if isinstance(motion_files, (str, bytes)) or not motion_files:
        raise ValueError("motion_files must be a nonempty sequence of NPZ paths.")
    files = list(motion_files)
    command = getattr(env_cfg.commands, command_name)
    command.motion_files = files
    configure_scene_motion_paired_terrain(env_cfg.scene, files)
    for event in vars(env_cfg.events).values():
        if getattr(event, "mode", None) != "reset":
            continue
        if event.params.get("command_name") == command_name and "pose_range" in event.params:
            event.params["pose_range"] = dict(event.params["pose_range"], x=(0.0, 0.0), y=(0.0, 0.0), yaw=(0.0, 0.0))


def configure_scene_motion_paired_terrain(scene_cfg: InteractiveSceneCfg, motion_files: Sequence[str]) -> None:
    """Replace scene terrain with NPZ geometry for task execution or reference playback."""
    previous = scene_cfg.terrain
    if isinstance(previous, MotionPairedTerrainImporterCfg):
        scene_cfg.terrain = replace(previous, motion_files=list(motion_files))
        return
    scene_cfg.terrain = MotionPairedTerrainImporterCfg(
        prim_path=previous.prim_path,
        physics_material=previous.physics_material,
        visual_material=previous.visual_material,
        debug_vis=previous.debug_vis,
        disable_visual=previous.disable_visual,
        motion_files=list(motion_files),
    )
