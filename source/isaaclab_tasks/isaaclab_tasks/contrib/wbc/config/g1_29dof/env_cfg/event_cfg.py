# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.wbc.mdp import events


@configclass
class G1EventCfg:
    reset_robot = EventTermCfg(func=events.reset_from_reference, mode="reset")
