# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from pathlib import Path

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.wbc.mdp.commands import WholeBodyCommand

from ..robot_constants import BODY_NAMES, JOINT_NAMES


@configclass
class WholeBodyCommandCfg(CommandTermCfg):
    """Static targets resampled within a common foot stance, without resetting physical state."""

    class_type: type = WholeBodyCommand
    dataset_path: str = str(Path(__file__).resolve().parents[3] / "data" / "stance_poses.npz")
    joint_names: list[str] = JOINT_NAMES
    body_names: list[str] = BODY_NAMES
    resampling_time_range: tuple[float, float] = (4.0, 10.0)
    initial_max_joint_rms: float = 0.25
    tracking_grace_period: float = 2.0


@configclass
class G1CommandsCfg:
    """Command terms for the G1 whole-body task."""

    whole_body = WholeBodyCommandCfg()
