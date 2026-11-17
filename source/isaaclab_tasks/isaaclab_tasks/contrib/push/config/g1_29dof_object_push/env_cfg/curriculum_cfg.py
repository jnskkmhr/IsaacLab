# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep tracking tolerances fixed while validating multi-teacher distillation."""

from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    """No curriculum in the initial rigid-ground pushing task."""

    pass
