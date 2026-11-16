# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command configuration for the `box` two-way coupling task."""

from isaaclab.utils import configclass


@configclass
class BoxCommandsCfg:
    """Empty command manager (target support pose is supplied in observation/reward terms)."""
