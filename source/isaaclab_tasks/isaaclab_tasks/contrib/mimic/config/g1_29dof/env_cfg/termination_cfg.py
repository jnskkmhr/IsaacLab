# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp


@configclass
class G1TerminationsCfg:
    """Termination terms for the MDP."""

    # -- time out terms
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # The motion command holds at its last frame instead of resetting itself, so this is what ends
    # the episode when the clip runs out. `time_out=True`: finishing the reference is a success, so
    # it is bootstrapped rather than treated as a failure (by PPO and by adaptive sampling alike).
    end_of_reference = DoneTerm(
        func=mimic_mdp.end_of_reference,
        params={"command_name": "motion"},
        time_out=True,
    )

    # -- poor tracking terms
    anchor_pos = DoneTerm(
        func=mimic_mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mimic_mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mimic_mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )

    # -- falls, including during a stance interval where the tracking-error terms above are gated
    # off. Measured base-to-lowest-foot so it stays valid on the uneven/soft terrain.
    base_too_low = DoneTerm(
        func=mimic_mdp.root_height_below_minimum_adaptive,
        params={
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # -- extreme base velocity term
    base_ang_vel_exceed = DoneTerm(
        func=mimic_mdp.base_ang_vel_exceed,
        params={"threshold": 500 * math.pi / 180.0},
    )
