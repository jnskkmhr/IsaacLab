# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.wbc.mdp as mdp


@configclass
class G1EventCfg:
    """Episode initialization and ungated interval perturbations."""

    reset_robot = EventTerm(func=mdp.reset_from_reference, mode="reset")
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )
    push_foot = EventTerm(
        func=mdp.push_body,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "force_range": {"x": (-100.0, 100.0), "y": (-100.0, 100.0), "z": (0.0, 0.0)},
            "torque_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_.*"]),
        },
    )
