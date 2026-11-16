# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Freely moving Unitree Go2 with implicit PD joint drives over a finite sand bed."""

from isaaclab_newton.sim.schemas import NewtonCollisionCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from particle_nerd_benchmark.core.terrain import configure_terrain


@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    robot = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = False
    robot.spawn.collision_props = {r"/.*foot/.*": [NewtonCollisionCfg(contact_margin=0.02, contact_gap=0.0)]}
    robot.init_state.pos = (-0.35, 0.0, 0.34)
    robot.actuators = {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            joint_effort_limit={".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 45.43},
            joint_velocity_limit=15.0,
            stiffness=100.0,
            damping=5.0,
        )
    }

    def __post_init__(self):
        configure_terrain(self, "flat", (2.0, 1.2, 0.30))
