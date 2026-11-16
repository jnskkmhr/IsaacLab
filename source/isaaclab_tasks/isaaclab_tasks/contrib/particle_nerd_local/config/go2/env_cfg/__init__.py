# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration submodules for the `go2` quadruped granular locomotion task."""

from .action_cfg import Go2ActionsCfg
from .commands_cfg import Go2CommandsCfg
from .curriculum_cfg import Go2CurriculumCfg
from .event_cfg import Go2EventsCfg
from .observation_cfg import Go2ObservationsCfg
from .physics_cfg import make_mpm_coupler_sim_cfg, make_particle_nerd_local_sim_cfg
from .reward_cfg import Go2RewardsCfg
from .scene_cfg import Go2SceneCfg
from .termination_cfg import Go2TerminationsCfg

__all__ = [
    "Go2ActionsCfg",
    "Go2CommandsCfg",
    "Go2CurriculumCfg",
    "Go2EventsCfg",
    "Go2ObservationsCfg",
    "Go2RewardsCfg",
    "Go2SceneCfg",
    "Go2TerminationsCfg",
    "make_mpm_coupler_sim_cfg",
    "make_particle_nerd_local_sim_cfg",
]
