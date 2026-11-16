# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat mixed rigid/Particle NeRD terrain with sustained torso-force disturbances."""

from isaaclab.utils import configclass

from .env_cfg.event_cfg import G1FlatForceEventCfg
from .env_cfg.terrain_cfg import FLAT_TERRAINS_CFG
from .mixed_env_cfg import G1MixedTerrainEnvCfg


@configclass
class G1FlatMixedTerrainEnvCfg(G1MixedTerrainEnvCfg):
    """Preserve WBC learning settings while removing terrain height variation."""

    events: G1FlatForceEventCfg = G1FlatForceEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG.replace(num_cols=2)
        self.curriculum.terrain_levels = None


@configclass
class G1FlatMixedTerrainEnvCfg_PLAY(G1FlatMixedTerrainEnvCfg):
    """Evaluate on one rigid and one particle environment, retaining force pulses."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.observations.policy.enable_corruption = False
        self.video_recorders = []
