# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-hand G1 pushes a randomly selected object across a finite, fully simulated sand bed."""

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg
from isaaclab.cloner.cloner_strategies import random
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsMeshCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import UNITREE_G1_29DOF_WITH_HAND_CFG

OBJECT_SIZE = (0.65, 0.65, 0.65)

BED_SIZE = (3.0, 2.0, 0.3)
VOXEL_SIZE = 0.04
FOOT_CONTACT_MARGIN = 0.5 * VOXEL_SIZE
"""MPM foot collider margin [m]."""


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
# Nominal PD gains: legs/waist from locomotion hpmmpox5; arms from manipulation 6pbvl2df.
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
    "waist": {"stiffness": 28.50124619574858, "damping": 1.814445686584846},
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
    """G1 with closed hands, a randomly selected object, and an unconfined sand grid on flat ground."""

    clone_cfg = CloneCfg(clone_strategy=random)

    robot: ArticulationCfg = ROBOT_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ROBOT_CFG.init_state.replace(pos=(-0.9, 0.0, 0.78)),
        spawn=UNITREE_G1_29DOF_WITH_HAND_CFG.spawn.replace(  # type: ignore
            # Couple the authored foot boxes to the sand with an MPM contact margin.
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

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, 0.0, OBJECT_SIZE[2] / 2 + 0.01)),
        spawn=sim_utils.MultiAssetSpawnerCfg(
            random_choice=False,  # Sampling is controlled by clone_cfg above.
            assets_cfg=[
                sim_utils.MeshCuboidCfg(size=OBJECT_SIZE),
                sim_utils.MeshSphereCfg(radius=OBJECT_SIZE[0] / 2),
                sim_utils.MeshCylinderCfg(radius=OBJECT_SIZE[0] / 2, height=OBJECT_SIZE[2], axis="Z"),
            ],
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=UsdPhysicsRigidBodyCfg(),
            collision_props=[UsdPhysicsCollisionCfg(), NewtonCollisionCfg(contact_margin=0.005, contact_gap=0.0)],
        ),
    )
    # MJWarp requires the same collider type in every world: use convex meshes for all variants.
    for shape in object.spawn.assets_cfg:
        shape.mesh_collision_props = UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")
        shape.physics_material = RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5)
        shape.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.75))
    del shape

    sand = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMGridCfg(
            visible=True,
            lower=(-BED_SIZE[0] / 2, -BED_SIZE[1] / 2, -BED_SIZE[2]),
            upper=(BED_SIZE[0] / 2, BED_SIZE[1] / 2, 0.0),
            voxel_size=VOXEL_SIZE,
            particles_per_cell=1.25,
            particle_placement="cell_center",
            jitter=0.05,
            visual_color=(0.72, 0.55, 0.34),
            material=MPMParticleMaterialCfg(
                density=1600.0,
                young_modulus=1.0e15,
                poisson_ratio=0.3,
                friction=0.58,
                yield_pressure=1.0e15,
                tensile_yield_ratio=0.0,
                yield_stress=0.0,
                hardening=0.0,
                dilatancy=0.0,
                viscosity=0.0,
            ),
        ),
    )
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        collision_group=-1,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -BED_SIZE[2])),
        spawn=sim_utils.GroundPlaneCfg(
            size=(1000.0, 1000.0),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.6, dynamic_friction=0.5),
        ),
    )
