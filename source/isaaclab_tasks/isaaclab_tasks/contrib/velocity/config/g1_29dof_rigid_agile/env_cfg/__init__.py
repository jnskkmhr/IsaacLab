# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .action_cfg import G1ActionsCfg
from .observation_cfg import G1ObservationsCfg
from .physics_cfg import G1PhysicsCfg
from .reward_cfg import G1RewardsCfg, G1RewardsRoughCfg
from .scene_cfg import G1SceneCfg
from .termination_cfg import G1TerminationsCfg
from .curriculum_cfg import G1CurriculumCfg
from .event_cfg import G1EventCfg
from .commands_cfg import G1CommandsCfg

from .terrain_cfg import (
    FLAT_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
    SLOPE_TERRAINS_CFG,
    WAVE_TERRAINS_CFG,
)
