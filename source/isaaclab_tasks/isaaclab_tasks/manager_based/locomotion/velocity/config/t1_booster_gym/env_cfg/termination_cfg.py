# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

@configclass
class T1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact, 
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=[".*Trunk", ".*Hip.*", ".*Shank.*"]
            ), 
            "threshold": 1.0},
    )
    base_too_low = DoneTerm(
        func=vel_mdp.root_height_below_minimum_adaptive,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 0.5
        }
    )