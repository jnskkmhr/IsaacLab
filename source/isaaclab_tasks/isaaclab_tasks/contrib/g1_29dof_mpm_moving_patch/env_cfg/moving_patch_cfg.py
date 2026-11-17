# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body selections and region/working-set settings for native moving MPM activity."""

from isaaclab.utils.configclass import configclass


@configclass
class BodyRegionCfg:
    """World-axis-aligned region following each body matched by the expression."""

    body_path: str = r"/World/envs/env_[^/]+/Robot/(?:.*/)?pelvis"
    half_extents: tuple[float, float, float] = (0.8, 0.8, 0.8)
    """XYZ half sizes [m], excluding Newton's additional stationary support halo."""
    offset: tuple[float, float, float] = (0.0, 0.0, -0.5)
    """World-frame offset [m] from the tracked body position to the region center."""


@configclass
class MovingPatchCfg:
    """Combine tracked regions, then apply Newton's fine-grained activity selection."""

    regions: list[BodyRegionCfg] = [BodyRegionCfg()]
    keep_material_motion: bool = True
    """Let moving/changing material outside tracked regions settle before sleeping.

    False imposes a hard region cutoff and can freeze moving sand. This is an approximation,
    not a guaranteed equivalent simulation. Overlapping regions never duplicate particles.
    """
    working_set_capacity: int | None = None
    """Evaluation-particle capacity; None reserves the full bed for safe reset/motion coverage.

    Only awake/support particles are evaluated. Full-bed particle storage remains resident.
    Smaller capacities are explicit limits: overflow raises instead of dropping particles.
    """
    working_set_page_size: int = 0
    """Zero compacts individual particle IDs; positive values compact original-ID pages."""
