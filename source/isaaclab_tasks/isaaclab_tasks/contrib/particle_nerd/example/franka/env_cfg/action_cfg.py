# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for the `franka` sand-pushing task."""

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils import configclass


@configclass
class FrankaActionsCfg:
    """7-DoF arm joint-position action offsets around the sand-immersed initial posture."""

    arm_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        scale=0.25,
        use_default_offset=True,
    )
