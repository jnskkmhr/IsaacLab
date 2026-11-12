# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import LegacyRayCaster

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact.mdp.rough_terrain import CurriculumSoftRoughTerrainCfg
from isaaclab_tasks.contrib.velocity.config.vel_mdp.rough_height import elevated_grid_pattern

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from .terrain_cfg import ROUGH_TERRAINS_CFG


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/unitree/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0,
                angular_damping=0,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                retain_accelerations=False,
                max_depenetration_velocity=1,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0, 0, 0.76],
            rot=(0.0, 0.0, 0.0, 1.0),
            ang_vel=[0, 0, 0],
            lin_vel=[0, 0, 0],
            joint_pos={
                ".*_knee_joint": 0.669,
                ".*_elbow_joint": 0.6,
                ".*_hip_pitch_joint": -0.312,
                ".*_ankle_pitch_joint": -0.363,
                "left_shoulder_roll_joint": 0.2,
                "left_shoulder_pitch_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_pitch_joint": 0.2,
            },
            joint_vel={".*": 0},
        ),
        actuators={
            "arms": ImplicitActuatorCfg(
                damping={
                    ".*_elbow_joint": 0.907222843292423,
                    ".*_wrist_yaw_joint": 1.06814150219,
                    ".*_wrist_roll_joint": 0.907222843292423,
                    ".*_wrist_pitch_joint": 1.06814150219,
                    ".*_shoulder_yaw_joint": 0.907222843292423,
                    ".*_shoulder_roll_joint": 0.907222843292423,
                    ".*_shoulder_pitch_joint": 0.907222843292423,
                },
                armature={
                    ".*_elbow_joint": 0.003609725,
                    ".*_wrist_yaw_joint": 0.00425,
                    ".*_wrist_roll_joint": 0.003609725,
                    ".*_wrist_pitch_joint": 0.00425,
                    ".*_shoulder_yaw_joint": 0.003609725,
                    ".*_shoulder_roll_joint": 0.003609725,
                    ".*_shoulder_pitch_joint": 0.003609725,
                },
                stiffness={
                    ".*_elbow_joint": 14.25062309787429,
                    ".*_wrist_yaw_joint": 16.77832748089279,
                    ".*_wrist_roll_joint": 14.25062309787429,
                    ".*_wrist_pitch_joint": 16.77832748089279,
                    ".*_shoulder_yaw_joint": 14.25062309787429,
                    ".*_shoulder_roll_joint": 14.25062309787429,
                    ".*_shoulder_pitch_joint": 14.25062309787429,
                },
                effort_limit_sim={
                    ".*_elbow_joint": 25,
                    ".*_wrist_yaw_joint": 5,
                    ".*_wrist_roll_joint": 25,
                    ".*_wrist_pitch_joint": 5,
                    ".*_shoulder_yaw_joint": 25,
                    ".*_shoulder_roll_joint": 25,
                    ".*_shoulder_pitch_joint": 25,
                },
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
                velocity_limit_sim={
                    ".*_elbow_joint": 37,
                    ".*_wrist_yaw_joint": 22,
                    ".*_wrist_roll_joint": 37,
                    ".*_wrist_pitch_joint": 22,
                    ".*_shoulder_yaw_joint": 37,
                    ".*_shoulder_roll_joint": 37,
                    ".*_shoulder_pitch_joint": 37,
                },
            ),
            "feet": ImplicitActuatorCfg(
                damping=1.814445686584846,
                armature=0.00721945,
                stiffness=28.50124619574858,
                effort_limit_sim=50,
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                velocity_limit_sim=37,
            ),
            "legs": ImplicitActuatorCfg(
                damping={
                    ".*_knee_joint": 6.3088018534966395,
                    ".*_hip_yaw_joint": 2.5578897650279457,
                    ".*_hip_roll_joint": 6.3088018534966395,
                    ".*_hip_pitch_joint": 2.5578897650279457,
                },
                armature={
                    ".*_knee_joint": 0.025101925,
                    ".*_hip_yaw_joint": 0.01017752,
                    ".*_hip_roll_joint": 0.025101925,
                    ".*_hip_pitch_joint": 0.01017752,
                },
                stiffness={
                    ".*_knee_joint": 99.09842777666113,
                    ".*_hip_yaw_joint": 40.17923847137318,
                    ".*_hip_roll_joint": 99.09842777666113,
                    ".*_hip_pitch_joint": 40.17923847137318,
                },
                effort_limit_sim={
                    ".*_knee_joint": 139,
                    ".*_hip_yaw_joint": 88,
                    ".*_hip_roll_joint": 139,
                    ".*_hip_pitch_joint": 88,
                },
                joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
                velocity_limit_sim={
                    ".*_knee_joint": 20,
                    ".*_hip_yaw_joint": 32,
                    ".*_hip_roll_joint": 20,
                    ".*_hip_pitch_joint": 32,
                },
            ),
            "waist": ImplicitActuatorCfg(
                damping=1.814445686584846,
                armature=0.00721945,
                stiffness=28.50124619574858,
                effort_limit_sim=50,
                joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
                velocity_limit_sim=37,
            ),
            "waist_yaw": ImplicitActuatorCfg(
                damping=2.5578897650279457,
                armature=0.01017752,
                stiffness=40.17923847137318,
                effort_limit_sim=88,
                joint_names_expr=["waist_yaw_joint"],
                velocity_limit_sim=32,
            ),
        },
        soft_joint_pos_limit_factor=0.9,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        max_init_terrain_level=0,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=10,
            num_cols=10,
            border_width=0.0,
            curriculum=True,
            use_cache=False,
            sub_terrains={
                "hard_ground": MeshPlaneTerrainCfg(size=(8.0, 8.0), proportion=0.5, ground_height_range=(0.0, 0.0)),
                "soft_ground": MeshPlaneTerrainCfg(size=(8.0, 8.0), proportion=0.5, ground_height_range=(0.0, -0.12)),
            },
        ),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.005,
        force_threshold=1.0,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class HeightGridPatternCfg(patterns.GridPatternCfg):
    """Downward rays starting above the body in world Z [m]."""

    func = elevated_grid_pattern
    height: float = 20.0


@configclass
class G1RoughSceneCfg(G1SceneCfg):
    """Reference robot and sensors with terrain-relative height measurements."""

    terrain = CurriculumSoftRoughTerrainCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        max_init_terrain_level=0,
        terrain_generator=ROUGH_TERRAINS_CFG.replace(num_cols=10),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    height_scanner = RayCasterCfg(
        class_type=LegacyRayCaster,  # Query only the specified surface mesh, not buried support.
        prim_path="{ENV_REGEX_NS}/Robot/.*pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=HeightGridPatternCfg(resolution=0.1, size=(0.2, 0.2)),
        mesh_prim_paths=["/World/ground/terrain"],
        global_world_only=True,
    )
    left_foot_scanner = height_scanner.replace(
        prim_path="{ENV_REGEX_NS}/Robot/.*left_ankle_roll_link",
        pattern_cfg=HeightGridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
    )
    right_foot_scanner = left_foot_scanner.replace(prim_path="{ENV_REGEX_NS}/Robot/.*right_ankle_roll_link")
