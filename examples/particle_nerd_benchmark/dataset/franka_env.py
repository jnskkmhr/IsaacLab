# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka with a welded open-box shovel (floor, back wall and two side walls) pushing MPM sand.

The tool is wider and larger than the benchmark's filled paddle. Its open front faces +x, the push direction, so
sand collects in front of the back wall and spills over the side walls as the shovel advances.
"""

from isaaclab_newton.sim.schemas import NewtonCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass, replace

from ..config.robot.env_cfg.scene_cfg import _make_franka_paddle_robot_cfg, _spawn_fixed_paddle
from ..config.robot.franka_env_cfg import FrankaEnvCfg
from ..core.physics import physics_cfg
from ..core.terrain import configure_terrain

WALL = 0.012
LENGTH = 0.22
WIDTH = 0.28
HEIGHT = 0.10
FLOOR_Z = 0.171
"""Hand-frame z of the shovel floor centre; its bottom matches the paddle's bottom (0.165 m)."""
PLATES = {
    "ShovelFloor": ((LENGTH, WIDTH, WALL), (0.0, 0.0, FLOOR_Z), 0.40),
    "ShovelBack": ((WALL, WIDTH, HEIGHT), (-LENGTH / 2 + WALL / 2, 0.0, FLOOR_Z - HEIGHT / 2), 0.15),
    "ShovelLeft": ((LENGTH, WALL, HEIGHT), (0.0, WIDTH / 2 - WALL / 2, FLOOR_Z - HEIGHT / 2), 0.15),
    "ShovelRight": ((LENGTH, WALL, HEIGHT), (0.0, -WIDTH / 2 + WALL / 2, FLOOR_Z - HEIGHT / 2), 0.15),
}
"""Shovel plates: name -> (size, offset in the panda_hand frame, mass)."""


def _spawn_franka_with_shovel(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Author the four welded plates under `panda_hand` inside the robot's clone prototype."""
    robot_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = robot_prim.GetStage()
    for hand_prim in sim_utils.find_matching_prims(f"{prim_path}/.*/panda_hand", stage=stage):
        for name, (size, offset, mass) in PLATES.items():
            plate = replace(cfg.plate, size=size, mass_props=sim_utils.MassCfg(mass=mass))
            path = f"{hand_prim.GetPath()}/{name}"
            plate.func(path, plate, translation=offset)
            visual = replace(cfg.plate_visual, size=size)
            visual.func(f"{path}/Visual", visual)
    return robot_prim


@configclass
class FrankaWithShovelCfg(sim_utils.UsdFileCfg):
    """Franka Panda and a welded open-box shovel imported together."""

    func = _spawn_franka_with_shovel
    plate = sim_utils.CuboidCfg(
        func=_spawn_fixed_paddle,
        size=(WALL, WALL, WALL),
        rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
        mass_props=sim_utils.MassCfg(mass=0.15),
        collision_props=[
            UsdPhysicsCollisionCfg(collision_enabled=True),
            NewtonCollisionCfg(contact_margin=0.016, contact_gap=0.0),
        ],
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    plate_visual = sim_utils.CuboidCfg(
        size=(WALL, WALL, WALL),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.45, 0.82), metallic=0.25, roughness=0.35),
    )


def _shovel_robot_cfg():
    cfg = _make_franka_paddle_robot_cfg()
    old = cfg.spawn
    cfg.spawn = FrankaWithShovelCfg(
        usd_path=old.usd_path,
        variants=old.variants,
        rigid_props=old.rigid_props,
        articulation_props=old.articulation_props,
        joint_drive_props=old.joint_drive_props,
        activate_contact_sensors=False,
    )
    # Soft joint gains with a higher effort limit track best while the wide shovel plows through sand; stiffer gains
    # made the base joint lag more (the lagged sand coupling fights a stiff arm).
    cfg.actuators["panda_arm"].joint_effort_limit = 400.0
    cfg.actuators["panda_arm"].stiffness = 800.0
    cfg.actuators["panda_arm"].damping = 80.0
    return cfg


@configclass
class ShovelFrankaSceneCfg(InteractiveSceneCfg):
    robot = _shovel_robot_cfg()

    def __post_init__(self):
        configure_terrain(self, "flat", (1.2, 1.0, 0.30), wall_top=0.04)


@configclass
class ShovelFrankaActionsCfg:
    """Absolute joint position targets for the seven arm joints (the fingers keep their default target)."""

    arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint[1-7]"], scale=1.0, use_default_offset=False
    )


@configclass
class ShovelFrankaEnvCfg(FrankaEnvCfg):
    scene = ShovelFrankaSceneCfg(num_envs=1, env_spacing=4.0)
    actions = ShovelFrankaActionsCfg()
    episode_length_s = 10.0
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=physics_cfg([r"/World/envs/env_.*/Robot"], [r"/World/envs/env_.*/Robot/.*Shovel.*"]),
        use_newton_actuators=True,
    )
