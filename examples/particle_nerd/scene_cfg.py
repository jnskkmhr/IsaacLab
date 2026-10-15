# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small sand bed; every collider is represented by a rigid body with an object ID."""

import math

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass

VOXEL_SIZE = 0.04
BED_SIZE = (0.32, 0.32, 0.16)
PARTICLES_PER_CELL = 1.25
RESOLUTION = tuple(math.ceil(s * PARTICLES_PER_CELL / VOXEL_SIZE) for s in BED_SIZE)
PARTICLE_RADIUS = 0.5 * (math.prod(BED_SIZE) / math.prod(RESOLUTION)) ** (1.0 / 3.0)
# Box centers and full extents in the environment frame [m]. Initial sand surface: z = 0.
BODY_POSITIONS = ((0, 0, -0.18), (-0.18, 0, -0.06), (0.18, 0, -0.06), (0, -0.18, -0.06), (0, 0.18, -0.06), (0, 0, 0.08))
BODY_SIZES = (
    (0.40, 0.40, 0.04),
    (0.04, 0.40, 0.20),
    (0.04, 0.40, 0.20),
    (0.32, 0.04, 0.20),
    (0.32, 0.04, 0.20),
    (0.12, 0.12, 0.08),
)
BODY_FRICTION = 0.6
MOVING_BODY_ID = 5
MOVING_BODY_MASS = 2.0


def body_cfg(object_id: int) -> RigidObjectCfg:
    """Create a box; fixed and moving boxes use the same collider representation."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Body{object_id}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BODY_POSITIONS[object_id]),
        spawn=sim_utils.CuboidCfg(
            size=BODY_SIZES[object_id],
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=object_id != MOVING_BODY_ID),
            mass_props=sim_utils.MassCfg(mass=MOVING_BODY_MASS),
            collision_props=NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.0),
            physics_material=UsdPhysicsRigidBodyMaterialCfg(
                static_friction=BODY_FRICTION, dynamic_friction=BODY_FRICTION
            ),
        ),
    )


@configclass
class ParticleSceneCfg(InteractiveSceneCfg):
    """Complete sand bed, four retaining walls, rigid ground, and a freely moving box."""

    sand = MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMGridCfg(
            lower=(-0.16, -0.16, -0.16),
            upper=(0.16, 0.16, 0.0),
            voxel_size=VOXEL_SIZE,
            particles_per_cell=PARTICLES_PER_CELL,
            particle_placement="cell_center",
            jitter=0.0,
            radius=PARTICLE_RADIUS,
            material=MPMParticleMaterialCfg(
                density=1600.0,
                young_modulus=1.0e15,
                poisson_ratio=0.3,
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
    body0 = body_cfg(0)
    body1 = body_cfg(1)
    body2 = body_cfg(2)
    body3 = body_cfg(3)
    body4 = body_cfg(4)
    body5 = body_cfg(5)


def configure_collection_scene(cfg, scene: str, voxel_size: float):
    """Set sampling and a level tray; the G1 preset uses its tilting testbed dimensions."""
    if scene == "g1-testbed":
        length, width, depth = 2.0, 2.0, 0.3
        wall, floor = 0.1, 0.1
        positions = (
            (0, 0, -depth - floor / 2),
            (-(length + wall) / 2, 0, -depth / 2),
            ((length + wall) / 2, 0, -depth / 2),
            (0, -(width + wall) / 2, -depth / 2),
            (0, (width + wall) / 2, -depth / 2),
        )
        sizes = (
            (length + 2 * wall, width + 2 * wall, floor),
            (wall, width + 2 * wall, depth),
            (wall, width + 2 * wall, depth),
            (length, wall, depth),
            (length, wall, depth),
        )
        for i, (position, size) in enumerate(zip(positions, sizes, strict=True)):
            body = getattr(cfg.scene, f"body{i}")
            body.init_state.pos = position
            body.spawn.size = size
        cfg.scene.env_spacing = 4.0
    elif scene == "small":
        length, width, depth = BED_SIZE
    else:
        raise ValueError(f"Unknown collection scene: {scene}")
    sand = cfg.scene.sand.spawn
    sand.lower = (-length / 2, -width / 2, -depth)
    sand.upper = (length / 2, width / 2, 0.0)
    sand.voxel_size = voxel_size
    resolution = tuple(math.ceil(s * sand.particles_per_cell / voxel_size) for s in (length, width, depth))
    sand.radius = 0.5 * (length * width * depth / math.prod(resolution)) ** (1.0 / 3.0)
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.voxel_size = voxel_size
    return {
        "scene": scene,
        "bed_size_m": [length, width, depth],
        "lattice_resolution": list(resolution),
        "expected_particles_per_environment": math.prod(resolution),
        "particle_spacing_m": [s / n for s, n in zip((length, width, depth), resolution, strict=True)],
        "particle_radius_m": sand.radius,
        "particles_per_cell_axis": sand.particles_per_cell,
        "tray_motion": "fixed level tray",
    }


def configure_body_shape(cfg, shape: str, size):
    """Set moving-body geometry; dimensions are full local extents in x/y/z."""
    old = cfg.scene.body5.spawn
    common = dict(
        rigid_props=old.rigid_props,
        mass_props=old.mass_props,
        collision_props=old.collision_props,
        physics_material=old.physics_material,
    )
    if shape == "box":
        cfg.scene.body5.spawn = sim_utils.CuboidCfg(size=tuple(size), **common)
    elif shape == "sphere":
        cfg.scene.body5.spawn = sim_utils.SphereCfg(radius=size[0] / 2, **common)
    elif shape == "cylinder":
        cfg.scene.body5.spawn = sim_utils.CylinderCfg(radius=size[0] / 2, height=size[2], axis="Z", **common)
    else:
        raise ValueError(f"Unknown body shape: {shape}")
    cfg.body_shape = shape


def configured_body_sizes(cfg):
    """Return recorded full extents for all six colliders."""
    sizes = [getattr(cfg.scene, f"body{i}").spawn.size for i in range(5)]
    body = cfg.scene.body5.spawn
    if cfg.body_shape == "sphere":
        sizes.append((2 * body.radius,) * 3)
    elif cfg.body_shape == "cylinder":
        sizes.append((2 * body.radius, 2 * body.radius, body.height))
    else:
        sizes.append(body.size)
    return sizes
