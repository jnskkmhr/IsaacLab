# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for the `go2` quadruped granular locomotion task."""

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils import configclass


@configclass
class Go2ActionsCfg:
    """12-DoF leg joint-position action offsets around the nominal standing configuration."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )
