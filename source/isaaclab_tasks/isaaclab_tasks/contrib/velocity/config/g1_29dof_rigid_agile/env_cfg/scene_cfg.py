# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Adapted from Agile. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.robots.unitree import UNITREE_G1_29DOF_CFG

from ..mdp.actuators import DelayedDCMotorCfg, DelayedImplicitActuatorCfg
from .terrain_cfg import MEDIUM_ROUGH_TERRAIN_CFG


@configclass
class SelectiveSelfCollisionArticulationCfg(ArticulationCfg):
    """Apply task-scoped collision filters to Newton's imported model builder.

    G1's collision subtrees are USD instances, so their shapes cannot receive
    relationship or margin overrides after spawn. Newton exposes the populated
    model builder immediately before finalization; this hook applies the same
    shape-pair exclusions there without de-instancing or modifying upstream code.
    """

    self_collision_filter_body_pairs: tuple[tuple[str, str], ...] = ()
    """Body-name pairs whose collision shapes must not collide with each other."""

    newton_collision_shape_margin: float | None = None
    """Optional Newton margin applied to every shape belonging to this articulation."""

    def _post_spawn(self, stage: Any) -> None:
        super()._post_spawn(stage)
        if not self.self_collision_filter_body_pairs and self.newton_collision_shape_margin is None:
            return

        from isaaclab.physics import PhysicsEvent  # noqa: PLC0415
        from isaaclab.sim import SimulationContext  # noqa: PLC0415

        simulation_context = SimulationContext.instance()
        if simulation_context is None or not simulation_context.physics_manager.__name__.lower().startswith("newton"):
            return
        self._self_collision_root_name = (
            (self.spawn.spawn_path if self.spawn is not None and self.spawn.spawn_path is not None else self.prim_path)
            .rstrip("/")
            .rsplit("/", 1)[-1]
        )
        simulation_context.physics_manager.register_callback(
            self._apply_newton_self_collision_config,
            PhysicsEvent.MODEL_INIT,
            name=f"{self._self_collision_root_name}_selective_self_collision",
        )

    def _apply_newton_self_collision_config(self, _payload: Any) -> None:
        from isaaclab_newton.physics import NewtonBuilderCfg  # noqa: PLC0415

        from isaaclab.sim import SimulationContext  # noqa: PLC0415

        # Isaac Lab 3.0 shares the populated model builder through the simulation's backend registry.
        sim = SimulationContext.instance()
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        if builder is None:
            raise RuntimeError("Cannot apply selective self-collision filters: Newton builder is unavailable.")

        root_marker = f"/{self._self_collision_root_name}/"
        bodies: dict[tuple[int, str], int] = {}
        for body_index, (label, world) in enumerate(zip(builder.body_label, builder.body_world, strict=True)):
            if root_marker in label:
                bodies[(world, label.rsplit("/", 1)[-1])] = body_index
                if self.newton_collision_shape_margin is not None:
                    for shape_index in builder.body_shapes[body_index]:
                        builder.shape_margin[shape_index] = self.newton_collision_shape_margin

        worlds = sorted({world for world, _ in bodies})
        for world in worlds:
            for body_a_name, body_b_name in self.self_collision_filter_body_pairs:
                body_a = bodies.get((world, body_a_name))
                body_b = bodies.get((world, body_b_name))
                if body_a is None or body_b is None:
                    missing = [name for name, body in ((body_a_name, body_a), (body_b_name, body_b)) if body is None]
                    raise RuntimeError(
                        f"Cannot apply Newton self-collision filter in world {world}: missing body prim(s) {missing}."
                    )
                for shape_a in builder.body_shapes[body_a]:
                    for shape_b in builder.body_shapes[body_b]:
                        builder.add_shape_collision_filter_pair(shape_a, shape_b)


MAX_DELAY_PHY_STEPS = 4


MIN_DELAY_PHY_STEPS = 0


G1_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"


LEG_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]


ANKLE_JOINT_NAMES = [
    ".*_ankle_.*_joint",
]


FEET_LINK_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


WAIST_JOINT_NAMES = [
    "waist_.*_joint",
]


ARM_JOINT_NAMES = [
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]


HAND_JOINT_NAMES = [
    ".*_hand_.*",
]


DEFAULT_PELVIS_HEIGHT = 0.72


_COLLISION_FILTERS = tuple(
    pair
    for side in ("left", "right")
    for pair in (
        ("pelvis", f"{side}_hip_roll_link"),
        (f"{side}_knee_link", f"{side}_ankle_roll_link"),
        ("torso_link", f"{side}_shoulder_roll_link"),
        ("torso_link", f"{side}_shoulder_yaw_link"),
        (f"{side}_shoulder_pitch_link", f"{side}_shoulder_yaw_link"),
        (f"{side}_elbow_link", f"{side}_wrist_pitch_link"),
        (f"{side}_wrist_roll_link", f"{side}_wrist_yaw_link"),
    )
)


G1_29DOF_DELAYED_DC_MOTOR = SelectiveSelfCollisionArticulationCfg(
    self_collision_filter_body_pairs=_COLLISION_FILTERS,
    newton_collision_shape_margin=0.0,
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD_PATH,
        # Disable hands to accelerate training.
        variants={"Physics": "PhysX", "left_hand": "None", "right_hand": "None"},
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedDCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            joint_effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            joint_velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.02,
                ".*_knee_joint": 0.02,
            },
            saturation_effort=180.0,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "feet": DelayedDCMotorCfg(
            joint_names_expr=ANKLE_JOINT_NAMES,
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            joint_effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            joint_velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.02,
            saturation_effort=80.0,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "waist": DelayedDCMotorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            joint_effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            joint_velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 300.0,
                "waist_roll_joint": 300.0,
                "waist_pitch_joint": 300.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.02,
            saturation_effort=120.0,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "arms": DelayedDCMotorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            joint_effort_limit={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            joint_velocity_limit={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 90.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 60.0,
                ".*_wrist_.*_joint": 4.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 0.4,
                ".*_elbow_joint": 1.0,
                ".*_wrist_.*_joint": 0.2,
            },
            armature={
                ".*_shoulder_.*": 0.02,
                ".*_elbow_.*": 0.02,
                ".*_wrist_.*_joint": 0.02,
            },
            saturation_effort=40.0,
            min_delay=0,
            max_delay=0,
        ),
    },
)


# Keep the locomotion robot geometry and pose; use BeyondMimic's delayed implicit drives.
G1_29DOF_BeyondMimic_delay = G1_29DOF_DELAYED_DC_MOTOR.replace(
    actuators={
        name: DelayedImplicitActuatorCfg(
            joint_names_expr=cfg.joint_names_expr,
            joint_effort_limit=cfg.effort_limit_sim,
            joint_velocity_limit=cfg.velocity_limit_sim,
            stiffness=cfg.stiffness,
            damping=cfg.damping,
            armature=cfg.armature,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        )
        for name, cfg in UNITREE_G1_29DOF_CFG.actuators.items()
    }
)


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane" or "generator"
        terrain_generator=MEDIUM_ROUGH_TERRAIN_CFG,  # None or LESS_ROUGH_TERRAIN_CFG, MEDIUM_ROUGH_TERRAIN_CFG
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                f"TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot = G1_29DOF_BeyondMimic_delay.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )

    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
        global_world_only=True,
    )

    height_measurement_sensor_left_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.01, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )
    height_measurement_sensor_right_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.01, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )

    smooth_interval_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link", update_period=0.0, history_length=5, track_air_time=False
    )
