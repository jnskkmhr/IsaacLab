# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass

import isaaclab_tasks.contrib.wbc.mdp as mdp

from .observation_cfg import BODY_NAMES, JOINT_NAME


@configclass
class G1CommandsCfg:
    """Command terms for the G1 whole-body task."""

    whole_body = mdp.WholeBodyCommandCfg(joint_names=JOINT_NAME, body_names=BODY_NAMES)
