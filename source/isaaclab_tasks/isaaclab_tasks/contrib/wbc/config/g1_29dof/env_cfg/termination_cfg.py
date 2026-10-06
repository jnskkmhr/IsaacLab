# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.wbc.mdp as mdp


@configclass
class G1TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    end_of_reference = TerminationTermCfg(func=mdp.reference_finished, time_out=True)
    fallen = TerminationTermCfg(func=mdp.fallen)
    tracking_lost = TerminationTermCfg(func=mdp.tracking_lost)
