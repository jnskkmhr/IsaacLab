# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.utils import configclass

from ..robot_constants import JOINT_NAMES


@configclass
class G1ActionsCfg:
    joint_position = JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
    )
