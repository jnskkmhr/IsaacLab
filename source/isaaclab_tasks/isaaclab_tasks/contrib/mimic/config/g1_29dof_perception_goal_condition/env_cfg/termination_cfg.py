# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp

from .. import mdp as goal_mdp


@configclass
class G1TerminationsCfg:
    """Termination terms for the MDP."""

    # -- time out terms
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Only imitation episodes time out at clip end. This is not task success.
    end_of_reference = DoneTerm(
        func=goal_mdp.end_of_reference,
        params={"command_name": "motion"},
        time_out=True,
    )

    # -- poor tracking terms
    anchor_pos = DoneTerm(
        func=goal_mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=goal_mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=goal_mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                # "left_wrist_yaw_link",
                # "right_wrist_yaw_link",
            ],
        },
    )

    # -- falls, measured from the base to the lowest foot.
    base_too_low = DoneTerm(
        func=mimic_mdp.root_height_below_minimum_adaptive,
        params={
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    fallen = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]), "threshold": 1.0},
    )

    extreme_action = DoneTerm(
        func=mimic_mdp.ExtremeJointPositionAction,  # type: ignore
        params={
            "action_name": "joint_pos",
            "torque_fraction": 5.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
