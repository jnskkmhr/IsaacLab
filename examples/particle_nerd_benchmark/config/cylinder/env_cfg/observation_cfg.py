# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from particle_nerd_benchmark.core.mdp.observations import object_pose


@configclass
class ObservationsCfg:
    @configclass
    class StateCfg(ObservationGroupCfg):
        pose = ObservationTermCfg(func=object_pose)
        enable_corruption = False

    state = StateCfg()
