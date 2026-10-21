# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "capped_foot_clearance_reward",
    "compute_standing_contact_penalty",
    "ExtremeJointPositionAction",
    "foot_clearance_reward",
    "feet_edge_penalty",
    "MirrorActionTermCfg",
    "MirrorAugmentation",
    "MirrorJointPositionActionCfg",
    "MirrorObservationTermCfg",
    "compute_mirrored_states",
    "mirror_identity",
    "mirror_joints",
    "mirror_quat",
    "mirror_vec3",
]

from .rewards import capped_foot_clearance_reward, compute_standing_contact_penalty, feet_edge_penalty, foot_clearance_reward
from .symmetry import (
    MirrorActionTermCfg,
    MirrorAugmentation,
    MirrorJointPositionActionCfg,
    MirrorObservationTermCfg,
    compute_mirrored_states,
    mirror_identity,
    mirror_joints,
    mirror_quat,
    mirror_vec3,
)
from .terminations import ExtremeJointPositionAction
