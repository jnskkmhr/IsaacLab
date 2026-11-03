# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Adapted from Agile. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp as g1_mdp


@configclass
class G1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "limit_angle": math.radians(30.0),
        },
    )

    illegal_contacts = DoneTerm(
        func=g1_mdp.illegal_ground_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis", "torso_link"],
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
            "threshold": 20.0,
            "min_height": 0.45,
        },
    )

    illegal_base_height = DoneTerm(
        func=g1_mdp.illegal_base_height,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "height_threshold": 0.72 - 0.3,
        },
    )
