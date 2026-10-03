# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass

import isaaclab_tasks.contrib.wbc.mdp as mdp

from .observation_cfg import JOINT_NAME


@configclass
class G1ActionsCfg:
    joint_position = mdp.MirrorJointPositionActionCfg(
        asset_name="robot",
        mirror=mdp.mirror_g1_joints,
        mirror_params={"joint_names": JOINT_NAME},
        joint_names=JOINT_NAME,
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
    )
