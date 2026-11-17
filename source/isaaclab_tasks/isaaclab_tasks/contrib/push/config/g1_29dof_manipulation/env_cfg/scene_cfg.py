# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legless G1 suspended at its fixed pelvis with actuated waist joints."""

from isaaclab_newton.sim.schemas import NewtonArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR, UNITREE_G1_29DOF_WITH_HAND_CFG

UPPER_BODY_CFG = UNITREE_G1_29DOF_WITH_HAND_CFG.copy()
UPPER_BODY_CFG.spawn.usd_path = (
    f"{ISAACLAB_ASSETS_DATA_DIR}/unitree/g1_upper_body_with_hand/g1_upper_body_with_hand.usda"
)
UPPER_BODY_CFG.spawn.articulation_props = [NewtonArticulationCfg(self_collision_enabled=True)]
UPPER_BODY_CFG.spawn.fix_root_link = True
UPPER_BODY_CFG.spawn.activate_contact_sensors = False
UPPER_BODY_CFG.actuators = {name: UPPER_BODY_CFG.actuators[name] for name in ("arms", "waist", "waist_yaw", "hands")}

# # Double PD gains for all arm and waist joints relative to the shared robot configuration.
# for actuator in UPPER_BODY_CFG.actuators.values():
#     actuator.stiffness = (
#         {joint_expr: 2.0 * value for joint_expr, value in actuator.stiffness.items()}
#         if isinstance(actuator.stiffness, dict)
#         else 2.0 * actuator.stiffness
#     )
#     actuator.damping = (
#         {joint_expr: 2.0 * value for joint_expr, value in actuator.damping.items()}
#         if isinstance(actuator.damping, dict)
#         else 2.0 * actuator.damping
#     )

# Double wrist and waist roll/pitch PD gains, retaining the other joint gains.
for name, actuator in UPPER_BODY_CFG.actuators.items():
    if name == "arms":
        actuator.stiffness = {
            joint_expr: value * (2.0 if "wrist" in joint_expr else 1.0)
            for joint_expr, value in actuator.stiffness.items()
        }
        actuator.damping = {
            joint_expr: value * (2.0 if "wrist" in joint_expr else 1.0)
            for joint_expr, value in actuator.damping.items()
        }
    elif name == "waist" or name == "waist_yaw":
        actuator.stiffness *= 2.0
        actuator.damping *= 2.0


UPPER_BODY_CFG.init_state.pos = (0.0, 0.0, 1.0)
CLOSED_HAND_JOINT_POS = {
    name: position for name, position in UPPER_BODY_CFG.init_state.joint_pos.items() if "_hand_" in name
}
UPPER_BODY_CFG.init_state.joint_pos = CLOSED_HAND_JOINT_POS | {
    "waist_.*_joint": 0.0,
    ".*_shoulder_pitch_joint": -0.42,
    "left_shoulder_roll_joint": 0.17,
    "right_shoulder_roll_joint": -0.17,
    "left_shoulder_yaw_joint": 0.13,
    "right_shoulder_yaw_joint": -0.13,
    ".*_elbow_joint": 1.0,
    "left_wrist_roll_joint": -0.15,
    "right_wrist_roll_joint": 0.15,
    ".*_wrist_pitch_joint": -0.56,
    "left_wrist_yaw_joint": -0.15,
    "right_wrist_yaw_joint": 0.15,
}


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = UPPER_BODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=750.0))
