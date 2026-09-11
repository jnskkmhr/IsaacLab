# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whole body control environments for legged robots."""

import os
import toml

# Conveniences to other module directories via relative paths
MOTION_TRACKING_EXT_DIR = os.path.abspath(os.path.join(__file__, "../"))
"""Path to the extension source directory."""

MOTION_TRACKING_DATA_DIR = os.path.join(MOTION_TRACKING_EXT_DIR, "data")
"""Path to the extension data directory."""

from .config import g1_29dof