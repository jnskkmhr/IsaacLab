# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline sampling settings for collision-checked wrist commands."""

from dataclasses import dataclass, field

from .action_cfg import UPPER_JOINTS
from .scene_cfg import CLOSED_HAND_JOINT_POS


@dataclass
class HandPoseDatasetCfg:
    """Task-specific joint/body selections and centimeter-scale dataset coverage."""

    left_joint_names: tuple[str, ...] = tuple(UPPER_JOINTS[:7])
    right_joint_names: tuple[str, ...] = tuple(UPPER_JOINTS[7:])
    mirror_joint_signs: tuple[float, ...] = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    reference_body_name: str = "torso_link"
    left_body_name: str = "left_wrist_yaw_link"
    right_body_name: str = "right_wrist_yaw_link"
    left_elbow_name: str = "left_elbow_link"
    right_elbow_name: str = "right_elbow_link"
    fixed_joint_positions: dict[str, float] = field(default_factory=lambda: dict(CLOSED_HAND_JOINT_POS))
    joint_limit_margin: float = 0.02
    collision_clearance: float = 0.002
    require_elbows_outward: bool = True
    voxel_size: float = 0.01
    samples_per_voxel: int = 8
    candidates: int = 2_000_000
    validation_candidates: int = 100_000
    seed: int = 42
