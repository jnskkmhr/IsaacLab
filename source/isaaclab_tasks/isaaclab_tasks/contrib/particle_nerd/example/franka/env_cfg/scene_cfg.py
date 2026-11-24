# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene configuration for the `franka` sand-pushing task (`1.2 m x 1.0 m x 0.25 m` sand bed) with welded paddle."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.assets import ArticulationCfg

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.g1_29dof_mpm.env_cfg.scene_cfg import MPM_VISUAL_COLOR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

VOXEL_SIZE = 0.04
SPACING = VOXEL_SIZE / 1.25
SAND_DENSITY = 1600.0
PARTICLE_JITTER = 0.20 * SPACING
PARTICLE_SEED = 42
BED_SIZE = (1.2, 1.0, 0.25)
PADDLE_SIZE = (0.14, 0.18, 0.12)
PADDLE_OFFSET = (0.0, 0.0, 0.105)
PADDLE_MASS = 0.5


def sand_grid_positions(size: tuple[float, float, float] = BED_SIZE, spacing: float = SPACING) -> list[list[float]]:
    """Build jittered particle positions of the sand bed (`N=9,176`)."""
    length, width, depth = size
    axes = [
        np.arange(-length / 2 + spacing / 2, length / 2, spacing),
        np.arange(-width / 2 + spacing / 2, width / 2, spacing),
        np.arange(-depth + spacing / 2, 0.2, spacing),
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = points[points[:, 2] < 0.0]
    rng = np.random.default_rng(PARTICLE_SEED)
    lower = np.maximum(points - PARTICLE_JITTER, (-length / 2, -width / 2, -depth))
    upper = np.minimum(points + PARTICLE_JITTER, (length / 2, width / 2, 0.0))
    points[:, :2] = rng.uniform(lower[:, :2], upper[:, :2])
    points[:, 2] = rng.uniform(lower[:, 2], upper[:, 2])
    return points.astype(np.float32).tolist()


def _tray_wall_cfg(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionCfg(contact_margin=0.016),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.25, 0.30, 0.33)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _spawn_fixed_paddle(
    prim_path: str,
    cfg: sim.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Spawn a mass-bearing box paddle welded to its parent rigid body (`panda_hand`)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: PLC0415

    @sim.clone
    def _spawn_one(
        prim_path: str,
        cfg: sim.CuboidCfg,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> Usd.Prim:
        paddle_prim = sim.spawn_cuboid(
            prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        stage = sim.get_current_stage()
        parent_prim = paddle_prim.GetParent()
        if not parent_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Paddle parent '{parent_prim.GetPath()}' is not a rigid body.")

        UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")).MakeInvisible()

        local_position = translation if translation is not None else (0.0, 0.0, 0.0)
        local_rotation = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
        joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/FixedJoint")
        joint.CreateBody0Rel().SetTargets([parent_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(prim_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_position))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(local_rotation[3], Gf.Vec3f(*local_rotation[:3])))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        return paddle_prim

    return _spawn_one(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)


def _spawn_franka_with_paddle(
    prim_path: str,
    cfg: FrankaWithPaddleCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Usd.Prim:
    """Author the welded box paddle under `panda_hand` inside the robot's clone prototype."""
    robot_prim = sim.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = robot_prim.GetStage()
    for hand_prim in sim.find_matching_prims(f"{prim_path}/.*/panda_hand", stage=stage):
        paddle_path = f"{hand_prim.GetPath()}/Paddle"
        cfg.paddle.func(paddle_path, cfg.paddle, translation=cfg.paddle_offset)
        cfg.paddle_visual.func(f"{paddle_path}/PaddleVisual", cfg.paddle_visual)
    return robot_prim


@configclass
class FrankaWithPaddleCfg(sim.UsdFileCfg):
    """Franka Panda and welded end-effector paddle imported together."""

    func = _spawn_franka_with_paddle
    paddle_offset: tuple[float, float, float] = PADDLE_OFFSET
    paddle = sim.CuboidCfg(
        func=_spawn_fixed_paddle,
        size=PADDLE_SIZE,
        rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
        mass_props=sim.MassCfg(mass=PADDLE_MASS),
        collision_props=[
            UsdPhysicsCollisionCfg(collision_enabled=True),
            NewtonCollisionCfg(contact_margin=0.016, contact_gap=0.0),
        ],
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    paddle_visual = sim.CuboidCfg(
        size=PADDLE_SIZE,
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.18, 0.45, 0.82), metallic=0.25, roughness=0.35),
    )


def _make_franka_paddle_robot_cfg() -> ArticulationCfg:
    cfg = replace(FRANKA_PANDA_HIGH_PD_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    old_spawn = cfg.spawn
    cfg.spawn = FrankaWithPaddleCfg(
        usd_path=old_spawn.usd_path,
        variants={"Physics": "mujoco"},
        rigid_props=old_spawn.rigid_props,
        articulation_props=old_spawn.articulation_props,
        joint_drive_props=[MujocoJointCfg(actuatorgravcomp=True)],
        activate_contact_sensors=False,
    )
    cfg.init_state.pos = (-0.72, 0.0, 0.05)
    cfg.init_state.joint_pos = {
        "panda_joint1": -0.5303,
        "panda_joint2": 0.7779,
        "panda_joint3": 0.0045,
        "panda_joint4": -2.3577,
        "panda_joint5": -0.1167,
        "panda_joint6": 2.8139,
        "panda_joint7": 0.3748,
        "panda_finger_joint.*": 0.0,
    }
    cfg.actuators["panda_arm"].joint_effort_limit = 200.0
    cfg.actuators["panda_arm"].stiffness = 1600.0
    cfg.actuators["panda_arm"].damping = 80.0
    cfg.actuators["panda_hand"].stiffness = 400.0
    cfg.actuators["panda_hand"].damping = 40.0
    return cfg


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """Scene for the `franka` sand-pushing task."""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot = _make_franka_paddle_robot_cfg()

    sand = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMPointsCfg(
            positions=sand_grid_positions(),
            radius=0.5 * SPACING,
            mass=SAND_DENSITY * (SPACING**3),
            visual_color=MPM_VISUAL_COLOR,
            material=MPMParticleMaterialCfg(
                density=SAND_DENSITY,
                young_modulus=1.0e15,
                poisson_ratio=0.30,
                friction=0.48,
                yield_pressure=1.0e15,
                tensile_yield_ratio=0.0,
                yield_stress=0.0,
                hardening=0.0,
                dilatancy=0.0,
                viscosity=0.0,
            ),
        ),
    )

    def __post_init__(self) -> None:
        length, width, depth = BED_SIZE
        wall_top = 0.04
        wall_height = depth + wall_top
        wall_center = (wall_top - depth) / 2.0

        self.floor = _tray_wall_cfg("Floor", (length + 0.2, width + 0.2, 0.20), (0.0, 0.0, -depth - 0.10))
        self.wall_x0 = _tray_wall_cfg(
            "WallX0", (0.10, width + 0.2, wall_height), (-length / 2 - 0.05, 0.0, wall_center)
        )
        self.wall_x1 = _tray_wall_cfg("WallX1", (0.10, width + 0.2, wall_height), (length / 2 + 0.05, 0.0, wall_center))
        self.wall_y0 = _tray_wall_cfg("WallY0", (length, 0.10, wall_height), (0.0, -width / 2 - 0.05, wall_center))
        self.wall_y1 = _tray_wall_cfg("WallY1", (length, 0.10, wall_height), (0.0, width / 2 + 0.05, wall_center))

        for name in ("floor", "wall_x0", "wall_x1", "wall_y0", "wall_y1"):
            mirror = deepcopy(getattr(self, name))
            mirror.prim_path = mirror.prim_path.replace("}/", "}/Rigid")
            mirror.spawn.visible = False
            setattr(self, "rigid_" + name, mirror)
