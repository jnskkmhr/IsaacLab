# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility imports for symmetry terms now owned by :mod:`isaaclab_contrib.mdp`."""

from isaaclab_contrib.mdp import (
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

__all__ = [
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
