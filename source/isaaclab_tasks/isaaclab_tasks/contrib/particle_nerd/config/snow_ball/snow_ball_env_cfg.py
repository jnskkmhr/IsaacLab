# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The WorldParticle snow-ball scene with the object NeRD solver."""

from isaaclab.utils import configclass

from ....particle_world_batch.config.snow_ball.snow_ball_env_cfg import SnowBallParticleWorldEnvCfg
from ..physics import use_object_nerd


@configclass
class SnowBallParticleNeRDEnvCfg(SnowBallParticleWorldEnvCfg):
    """Preserve the reference snow task, including its slope, snow material and ball velocity."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = use_object_nerd(self.sim.physics)
