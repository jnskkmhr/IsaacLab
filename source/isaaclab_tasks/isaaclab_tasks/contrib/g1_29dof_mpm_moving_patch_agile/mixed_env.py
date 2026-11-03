# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference mixed-terrain mechanics with Agile's coupled force-history rewards."""

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mixed_env import G1MixedTerrainEnv

from .mpm_env import G1MovingPatchAgileEnv


class G1MixedTerrainAgileEnv(G1MovingPatchAgileEnv, G1MixedTerrainEnv):
    """Keep the reference particle allocation, reset, margin, and contact handling."""
