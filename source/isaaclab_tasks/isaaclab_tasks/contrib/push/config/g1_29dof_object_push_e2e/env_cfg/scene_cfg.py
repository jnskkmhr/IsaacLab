# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""No-hand G1 and a dynamic box on a rigid flat ground plane."""

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

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND

BOX_SIZE = (1.0, 0.8, 1.0)
"""Box length along the push direction, width, and height [m].

The box is taller than the wrists in the pushing posture, so walking forward brings them
onto its rear face. The length keeps the tipping limit ``length / (2 * friction)`` above
that push height for sliding friction up to about 0.6.
"""

BOX_MASS = 10.0
"""Nominal box mass [kg]; a startup event resamples it per environment."""

FOOT_CONTACT_MARGIN = 0.0
"""Rigid-ground foot collider margin [m]."""

ARM_PUSH_POSTURE = {
    ".*_shoulder_pitch_joint": -0.60,
    "left_shoulder_roll_joint": 0.30,
    "right_shoulder_roll_joint": -0.30,
    "left_shoulder_yaw_joint": -0.20,
    "right_shoulder_yaw_joint": 0.20,
    ".*_elbow_joint": 0.80,
    "left_wrist_roll_joint": -0.07,
    "right_wrist_roll_joint": 0.07,
    ".*_wrist_pitch_joint": 0.27,
    "left_wrist_yaw_joint": -0.10,
    "right_wrist_yaw_joint": 0.10,
}
"""Default arm joint positions [rad]: both wrists held ahead of the chest as a bumper.

Taken from the collision-free wrist-pose dataset of the manipulation task. Actions are
offsets from this posture and the posture reward pulls the arms back to it.
"""

# Keep the leg and waist defaults of the asset; replace only the arm joints.
ROBOT_CFG = UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND.copy()
ROBOT_CFG.init_state.joint_pos = {
    expression: value
    for expression, value in ROBOT_CFG.init_state.joint_pos.items()
    if not any(part in expression for part in ("shoulder", "elbow", "wrist"))
} | ARM_PUSH_POSTURE


@configclass
class G1ObjectPushSceneCfg(InteractiveSceneCfg):
    """No-hand G1, a box, foot-ground contact sensing, and robot-box contact sensing."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ROBOT_CFG.spawn.replace(  # type: ignore
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

    # Every robot body filtered against the box: separates wrist pushing from body contact.
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.1, 0.0, BOX_SIZE[2] / 2 + 0.01)),
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            rigid_props=UsdPhysicsRigidBodyCfg(),
            collision_props=[UsdPhysicsCollisionCfg(), NewtonCollisionCfg(contact_margin=0.005, contact_gap=0.0)],
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.75)),
        ),
    )
