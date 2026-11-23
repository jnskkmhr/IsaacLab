# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event configuration for the `particle` task."""

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from ..mdp.observation import reset_particle_block


@configclass
class ParticleEventsCfg:
    """Restore the sand block on reset; the release velocity is set by the environment configuration."""

    reset_sand = EventTermCfg(func=reset_particle_block, mode="reset", params={"velocity": (0.0, 0.0, 0.0)})
