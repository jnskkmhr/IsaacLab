# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft contact force model based on 2D/3D resistive force theory (RFT).

The model is applied as an action term that composes the resistive wrench onto the
articulation bodies intruding into the granular terrain. See :file:`README.md` for references.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
