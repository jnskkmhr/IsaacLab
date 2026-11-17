# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""No curricula: reward tolerances and command ranges are fixed for the first pushing stage."""

from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    """Fixed settings; add box-mass or command-range curricula here once pushing is established."""
