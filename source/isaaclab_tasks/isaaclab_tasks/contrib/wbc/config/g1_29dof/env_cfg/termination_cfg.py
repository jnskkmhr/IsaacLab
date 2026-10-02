# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.wbc.mdp import terminations


@configclass
class G1TerminationsCfg:
    time_out = TerminationTermCfg(func=terminations.time_out, time_out=True)
    reference_finished = TerminationTermCfg(func=terminations.reference_finished, time_out=True)
    fallen = TerminationTermCfg(func=terminations.fallen)
    tracking_lost = TerminationTermCfg(func=terminations.tracking_lost)
