# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.hector.mdp as hector_mdp

@configclass
class HECTORTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True) # type: ignore
    base_contact = DoneTerm(
        func=mdp.illegal_contact, # type: ignore
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["trunk"]), "threshold": 1.0},
    )
    # base_too_low = DoneTerm(
    #     func=hector_mdp.root_height_below_minimum_adaptive,  # type: ignore
    #     params={
    #         "minimum_height": 0.4,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
    #     },
    # )