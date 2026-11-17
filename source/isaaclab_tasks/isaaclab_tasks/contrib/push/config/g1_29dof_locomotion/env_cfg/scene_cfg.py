# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full G1 on flat rigid ground; arms remain physical and actuated."""

from isaaclab_newton.sim.schemas import NewtonCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND

# Use unitree rl lab waist roll/pitch gains
UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND_increased_waist_gain = UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND.copy()
for name, actuator in UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND_increased_waist_gain.actuators.items():
    if name == "waist":
        actuator.stiffness = 40.0
        actuator.damping = 5.0

@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND_increased_waist_gain.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND_increased_waist_gain.spawn.replace(
            collision_props={r"/.*_ankle_roll_link/.*": [NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.0)]}
        ),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link", history_length=3, track_air_time=True
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=750.0))
