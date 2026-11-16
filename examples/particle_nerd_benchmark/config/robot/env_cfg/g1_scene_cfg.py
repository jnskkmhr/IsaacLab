# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 on a finite, flat MPM sand bed; no recycling or learned particle solver."""

from isaaclab_newton.sim.schemas import NewtonCollisionCfg

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG

from particle_nerd_benchmark.core.terrain import configure_terrain


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    robot = replace(UNITREE_G1_29DOF_BOX_FOOT_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.collision_props = {
        r"/.*_ankle_roll_link/.*": [NewtonCollisionCfg(contact_margin=0.02, contact_gap=0.0)]
    }
    robot.init_state.pos = (-1.2, 0.0, 0.76)

    def __post_init__(self):
        configure_terrain(self, "flat", (3.6, 1.6, 0.30))
