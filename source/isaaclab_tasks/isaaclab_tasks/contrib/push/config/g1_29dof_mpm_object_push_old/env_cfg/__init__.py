# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone configuration components for whole-body object pushing."""

from .action_cfg import G1ActionsCfg
from .commands_cfg import CommandsCfg
from .curriculum_cfg import CurriculumCfg
from .event_cfg import EventsCfg
from .observation_cfg import ObservationsCfg
from .physics_cfg import DEFAULT_PROXY_MASS_SCALE, G1PhysicsProxyCfg
from .reward_cfg import RewardsCfg
from .scene_cfg import G1ObjectPushSceneCfg
from .termination_cfg import TerminationsCfg
