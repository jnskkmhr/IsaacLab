# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as Term
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid import mdp as rigid_mdp


@configclass
class G1TerminationsCfg:
    time_out = Term(func=mdp.time_out, time_out=True)

    base_too_low = Term(
        func=rigid_mdp.root_height_below_minimum_adaptive,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*_ankle_roll_link", preserve_order=False),
            "minimum_height": 0.2,
        },
        time_out=False,
    )

    bad_orientation = Term(func=mdp.bad_orientation, params={"limit_angle": 0.8}, time_out=False)
