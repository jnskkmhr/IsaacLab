# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 and a dynamic box on a rigid flat ground plane."""

from isaaclab_newton.sim.schemas import NewtonCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import UNITREE_G1_29DOF_WITH_HAND_CFG

BOX_SIZE = (0.65, 0.65, 0.65)

FOOT_CONTACT_MARGIN = 0.0
"""Rigid-ground foot collider margin [m]."""


# Lower-body action offsets match the locomotion teacher; arm offsets match the reaching teacher.
ROBOT_CFG = UNITREE_G1_29DOF_WITH_HAND_CFG.copy()
LOCOMOTION_REFERENCE_JOINT_POS = dict(ROBOT_CFG.init_state.joint_pos)
MANIPULATION_REFERENCE_JOINT_POS = {
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
ROBOT_CFG.init_state.joint_pos = {
    expression: value
    for expression, value in LOCOMOTION_REFERENCE_JOINT_POS.items()
    if not any(part in expression for part in ("waist", "shoulder", "elbow", "wrist"))
} | MANIPULATION_REFERENCE_JOINT_POS
# Nominal PD gains: legs/waist from locomotion 2jiewizp; arms from manipulation 6pbvl2df.
for actuator_name, gains in {
    "feet": {"stiffness": 28.50124619574858, "damping": 1.814445686584846},
    "legs": {
        "stiffness": {
            ".*_knee_joint": 99.09842777666113,
            ".*_hip_yaw_joint": 40.17923847137318,
            ".*_hip_roll_joint": 99.09842777666113,
            ".*_hip_pitch_joint": 40.17923847137318,
        },
        "damping": {
            ".*_knee_joint": 6.3088018534966395,
            ".*_hip_yaw_joint": 2.5578897650279457,
            ".*_hip_roll_joint": 6.3088018534966395,
            ".*_hip_pitch_joint": 2.5578897650279457,
        },
    },
    "waist": {"stiffness": 40, "damping": 5},
    "waist_yaw": {"stiffness": 40.17923847137318, "damping": 2.5578897650279457},
    "arms": {
        "stiffness": {
            ".*_elbow_joint": 14.25062309787429,
            ".*_wrist_yaw_joint": 33.55665496178558,
            ".*_wrist_roll_joint": 28.50124619574858,
            ".*_wrist_pitch_joint": 33.55665496178558,
            ".*_shoulder_yaw_joint": 14.25062309787429,
            ".*_shoulder_roll_joint": 14.25062309787429,
            ".*_shoulder_pitch_joint": 14.25062309787429,
        },
        "damping": {
            ".*_elbow_joint": 0.907222843292423,
            ".*_wrist_yaw_joint": 2.13628300438,
            ".*_wrist_roll_joint": 1.814445686584846,
            ".*_wrist_pitch_joint": 2.13628300438,
            ".*_shoulder_yaw_joint": 0.907222843292423,
            ".*_shoulder_roll_joint": 0.907222843292423,
            ".*_shoulder_pitch_joint": 0.907222843292423,
        },
    },
}.items():
    ROBOT_CFG.actuators[actuator_name].stiffness = gains["stiffness"]
    ROBOT_CFG.actuators[actuator_name].damping = gains["damping"]


@configclass
class SceneCfg(InteractiveSceneCfg):
    """G1 with closed hands, a 10 kg cube, and rigid foot-contact sensing."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UNITREE_G1_29DOF_WITH_HAND_CFG.spawn.replace(  # type: ignore
            # Keep the rigid foot contact surface at the authored collider.
            collision_props={
                r"/.*_ankle_roll_link/.*": [
                    NewtonCollisionCfg(
                        contact_margin=FOOT_CONTACT_MARGIN,
                        contact_gap=0.0,
                    ),
                ],
            },
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )

    # Newton supports filtering all robot bodies against the box in each environment.
    box_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
        history_length=1,
        update_period=0.0,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(BOX_SIZE[0] / 2 + 0.27, 0.0, BOX_SIZE[2] / 2 + 0.01)),
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=UsdPhysicsRigidBodyCfg(),
            collision_props=[UsdPhysicsCollisionCfg(), NewtonCollisionCfg(contact_margin=0.005, contact_gap=0.0)],
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.75)),
        ),
    )
