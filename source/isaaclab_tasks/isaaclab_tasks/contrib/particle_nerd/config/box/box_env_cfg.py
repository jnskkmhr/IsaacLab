# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The WorldParticle box scene with the object NeRD solver."""

from isaaclab.utils import configclass

from ....particle_world_batch.config.box.box_env_cfg import BoxParticleWorldEnvCfg
from ..physics import use_object_nerd


@configclass
class BoxParticleNeRDEnvCfg(BoxParticleWorldEnvCfg):
    """Preserve the reference box task and replace only its learned material solver."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = use_object_nerd(self.sim.physics)
