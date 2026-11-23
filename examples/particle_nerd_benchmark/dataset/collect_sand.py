# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Removed: sand-only episodes are collected by `collect_objects` in the same coupled scene as the object episodes.

This file only remains because it could not be deleted automatically; delete it.
"""

raise ImportError(
    "dataset.collect_sand was removed: use `python -m particle_nerd_benchmark.dataset.collect_objects "
    "--category rest_bed|freefall` (same scene and solver settings as the object episodes)."
)
