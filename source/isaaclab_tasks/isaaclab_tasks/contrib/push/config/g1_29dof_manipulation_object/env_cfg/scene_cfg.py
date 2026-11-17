# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legless G1 suspended at its fixed pelvis with actuated waist joints."""

from isaaclab_newton.sim.schemas import NewtonArticulationCfg, NewtonCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR, UNITREE_G1_29DOF_WITH_HAND_CFG

UPPER_BODY_CFG = UNITREE_G1_29DOF_WITH_HAND_CFG.copy()
UPPER_BODY_CFG.spawn.usd_path = (
    f"{ISAACLAB_ASSETS_DATA_DIR}/unitree/g1_upper_body_with_hand/g1_upper_body_with_hand.usda"
)
UPPER_BODY_CFG.spawn.articulation_props = [NewtonArticulationCfg(self_collision_enabled=True)]
UPPER_BODY_CFG.spawn.fix_root_link = True
UPPER_BODY_CFG.spawn.activate_contact_sensors = True
UPPER_BODY_CFG.actuators = {name: UPPER_BODY_CFG.actuators[name] for name in ("arms", "waist", "waist_yaw", "hands")}

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


# The USD torso is 0.837 m above the articulation spawn transform.
# Place the torso at world z=1.0 m, beside the table.
UPPER_BODY_CFG.init_state.pos = (0.0, 0.0, 0.163)
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
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    # Table top at z=0.95 m. Its rear edge is clear of the fixed torso.
    platform = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.82, 0.0, 0.90)),
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.0, 0.10),
            collision_props=[UsdPhysicsCollisionCfg(), NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.0)],
            physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.44, 0.0, 1.052)),
        spawn=sim_utils.CuboidCfg(
            size=(0.24, 0.44, 0.20),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=UsdPhysicsRigidBodyCfg(),
            collision_props=[UsdPhysicsCollisionCfg(), NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.0)],
            physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.75)),
        ),
    )
    object_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=1,
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=750.0))
