# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Construct the physical configuration shared by Gym playback and dataset collection."""


def make_config(motion: str, terrain: str, *, geometry: str | None = None):
    """Resolve one benchmark case without changing its controller or material settings."""
    from isaaclab.managers import ActionTermCfg

    from .mdp.actions import BodyWrenchAction
    from .terrain import configure_terrain

    if geometry not in (None, "box", "sphere", "cylinder"):
        raise ValueError(f"Unknown object geometry: {geometry}")
    if geometry is not None and motion in ("g1", "quadruped", "franka"):
        raise ValueError("Object geometry cannot override a robot task")
    if geometry == "cylinder":
        from ..config.cylinder.cylinder_env_cfg import ObjectEnvCfg
    elif geometry == "sphere":
        from ..config.sphere.sphere_env_cfg import ObjectEnvCfg
    elif motion == "g1":
        from ..config.robot.g1_env_cfg import G1EnvCfg as ObjectEnvCfg
    elif motion == "quadruped":
        from ..config.robot.quadruped_env_cfg import QuadrupedEnvCfg as ObjectEnvCfg
    elif motion == "franka":
        from ..config.robot.franka_env_cfg import FrankaEnvCfg as ObjectEnvCfg
    elif motion == "rolling" and geometry is None:
        from ..config.sphere.sphere_env_cfg import ObjectEnvCfg
    else:
        from ..config.cube.cube_env_cfg import ObjectEnvCfg
    cfg = ObjectEnvCfg()
    bed_size = {"franka": (1.2, 1.0, 0.25), "g1": (3.6, 1.6, 0.30), "quadruped": (2.0, 1.2, 0.30)}.get(
        motion, (2.0, 1.2, 0.30)
    )
    configure_terrain(cfg.scene, terrain, bed_size, wall_top=0.04 if motion == "franka" else 0.20)
    if motion == "projectile":
        if geometry is None:
            cfg.scene.object.spawn.size = (0.26, 0.18, 0.05)
        cfg.scene.object.spawn.mass_props.mass = 10.0
        cfg.scene.object.spawn.rigid_props.kinematic_enabled = False
        cfg.scene.object.init_state.pos = (-0.55, 0.0, 0.35)
        cfg.actions.motion = ActionTermCfg(class_type=BodyWrenchAction, asset_name="object")
    return cfg
