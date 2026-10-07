# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for fixed pelvis and parent-relative upper-body goals."""

import math
from dataclasses import MISSING
from pathlib import Path

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass


@configclass
class PoseGoalCommandCfg(CommandTermCfg):
    """Sample independent endpoint goals; commands remain fixed between resamples."""

    class_type: str = "{DIR}.pose_goal_commands:PoseGoalCommand"
    dataset_path: str = str(Path(__file__).resolve().parents[1] / "data/stance_poses.npz")
    joint_names: list[str] = MISSING
    pelvis_name: str = "pelvis"
    torso_name: str = "torso_link"
    hand_names: list[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    foot_names: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
    resampling_time_range: tuple[float, float] = (8.0, 12.0)
    initial_distance: float = 0.15
    final_distance: float = 1.0
    initial_yaw_change: float = 0.25
    final_yaw_change: float = 1.57
    curriculum_steps: int = 24000
    standing_probability: float = 0.2
    initial_min_height: float = 0.55
    contact_threshold: float = 5.0
    position_tolerance: float = 0.05
    height_tolerance: float = 0.03
    orientation_tolerance: float = 0.1
    hand_position_tolerance: float = 0.03
    linear_velocity_tolerance: float = 0.1
    angular_velocity_tolerance: float = 0.2
    relative_velocity_tolerance: float = 0.1
    hold_duration: float = 1.0
    max_visualized_envs: int = 20

    def validate_config(self) -> None:
        """Reject invalid sampling ranges and error scales before runtime."""
        positive = (
            self.contact_threshold,
            self.position_tolerance,
            self.height_tolerance,
            self.orientation_tolerance,
            self.hand_position_tolerance,
            self.linear_velocity_tolerance,
            self.angular_velocity_tolerance,
            self.relative_velocity_tolerance,
            self.hold_duration,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("WBC contact, tolerance, and hold parameters must be positive")
        if not 0 <= self.standing_probability <= 1 or self.curriculum_steps < 0:
            raise ValueError("Invalid WBC standing probability or curriculum duration")
        if not all(math.isfinite(value) for value in (self.initial_distance, self.final_distance)):
            raise ValueError("WBC goal distance bounds must be finite")
        if not 0 <= self.initial_distance <= self.final_distance:
            raise ValueError("WBC distance range must be nonnegative and increasing")
        if not 0 <= self.initial_yaw_change <= self.final_yaw_change <= 3.141593:
            raise ValueError("WBC yaw range must be within [0, pi] and increasing")
        if self.max_visualized_envs < 1 or not math.isfinite(self.initial_min_height):
            raise ValueError("Invalid WBC visualization count or initial height")
        low, high = self.resampling_time_range
        if not math.isfinite(low) or not math.isfinite(high) or not 0 < low <= high:
            raise ValueError("WBC command hold times must be finite, positive, and increasing")
        if len(self.hand_names) != 2 or len(self.foot_names) != 2:
            raise ValueError("WBC requires left/right hand and foot names")
