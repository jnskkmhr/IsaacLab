# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration submodules for the `g1` humanoid granular locomotion task."""

from .action_cfg import G1ActionsCfg
from .commands_cfg import G1CommandsCfg
from .curriculum_cfg import G1CurriculumCfg
from .event_cfg import G1EventsCfg
from .observation_cfg import G1ObservationsCfg
from .physics_cfg import make_mpm_coupler_sim_cfg, make_particle_nerd_sim_cfg
from .reward_cfg import G1RewardsCfg
from .scene_cfg import G1SceneCfg
from .termination_cfg import G1TerminationsCfg

__all__ = [
    "G1ActionsCfg",
    "G1CommandsCfg",
    "G1CurriculumCfg",
    "G1EventsCfg",
    "G1ObservationsCfg",
    "G1RewardsCfg",
    "G1SceneCfg",
    "G1TerminationsCfg",
    "make_mpm_coupler_sim_cfg",
    "make_particle_nerd_sim_cfg",
]
