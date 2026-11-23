# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The scene that ``record.py`` simulates with MPM: a tray of sand and one rigid object.

- Sand bed: a jittered particle lattice below a terrain surface (:func:`sand_cfg`, :func:`surface_height`) in a
  tray of five static boxes (:func:`configure_terrain`).
- Physics: implicit MPM for the sand, MuJoCo Warp for the object, coupled by the lagged reaction of the sand on
  the object (:func:`physics_cfg`).
- Object: a box, sphere or cylinder that is either moved along a prescribed path
  (:class:`QuietObjectStateAction`) or free and pushed by a wrench (:class:`BodyWrenchAction`).

:func:`make_env_cfg` puts these together. Lengths are in metres.
"""

from copy import deepcopy

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonMPMManager,
    NewtonShapeCfg,
)
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import ActionTerm, ActionTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

VOXEL_SIZE = 0.04
"""Cell size of the MPM grid [m]."""
SPACING = VOXEL_SIZE / 1.25
"""Distance between neighbouring particles of the lattice [m]."""
DENSITY = 1600.0
"""Sand density [kg/m^3]."""
PARTICLE_JITTER = 0.20 * SPACING
"""Largest random offset of a particle from its lattice position, per axis [m]."""
PARTICLE_SEED = 42
OBJECT_BODIES = [r"/World/envs/env_.*/Object"]


def surface_height(x, terrain):
    """Height of the sand surface at ``x``, relative to the tray centre.

    Args:
        x: Position along the tray [m].
        terrain: ``flat``, ``wave``, or ``slope_p<deg>`` / ``slope_m<deg>`` (rises or falls along +x).
    """
    if terrain == "flat":
        return np.zeros_like(x)
    if terrain.startswith("slope_"):
        sign = 1.0 if terrain[6] == "p" else -1.0
        return sign * np.tan(np.radians(float(terrain[7:]))) * np.asarray(x)
    if terrain == "wave":
        return 0.06 * np.sin(2 * np.pi * np.asarray(x) / 1.2)
    raise ValueError(f"Unknown terrain: {terrain}")


def sand_cfg(terrain="flat", size=(2.0, 1.2, 0.30), *, seed=PARTICLE_SEED):
    """Sand bed: lattice positions below the terrain surface, each jittered inside the bed.

    Every coordinate is drawn uniformly within :data:`PARTICLE_JITTER` of its lattice position, cut off at the
    tray and at the sand surface. The seed makes the positions reproducible.
    """
    length, width, depth = size
    axes = [
        np.arange(-length / 2 + SPACING / 2, length / 2, SPACING),
        np.arange(-width / 2 + SPACING / 2, width / 2, SPACING),
        np.arange(-depth + SPACING / 2, 0.2, SPACING),
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = points[points[:, 2] < surface_height(points[:, 0], terrain)]
    rng = np.random.default_rng(seed)
    lower = np.maximum(points - PARTICLE_JITTER, (-length / 2, -width / 2, -depth))
    upper = np.minimum(points + PARTICLE_JITTER, (length / 2, width / 2, np.inf))
    points[:, :2] = rng.uniform(lower[:, :2], upper[:, :2])
    upper[:, 2] = np.minimum(upper[:, 2], surface_height(points[:, 0], terrain))
    points[:, 2] = rng.uniform(lower[:, 2], upper[:, 2])
    return MPMObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sand",
        spawn=MPMPointsCfg(
            positions=points.tolist(),
            radius=SPACING / 2,
            mass=DENSITY * SPACING**3,
            visual_color=(0.72, 0.59, 0.36),
            material=MPMParticleMaterialCfg(
                density=DENSITY,
                young_modulus=1e15,
                poisson_ratio=0.3,
                friction=0.48,
                yield_pressure=1e15,
                tensile_yield_ratio=0.0,
                yield_stress=0.0,
                hardening=0.0,
                dilatancy=0.0,
                viscosity=0.0,
            ),
        ),
    )


def tray_box(name, size, position):
    """One static box of the tray."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim.CuboidCfg(
            size=size,
            collision_props=NewtonCollisionCfg(contact_margin=0.016),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.25, 0.30, 0.33)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


def configure_terrain(scene, terrain, size=(2.0, 1.2, 0.30), *, wall_top=0.20):
    """Set the sand bed and the tray (a floor and four walls) of ``scene``."""
    length, width, depth = size
    scene.sand = sand_cfg(terrain, size)
    scene.floor = tray_box("Floor", (length + 0.2, width + 0.2, 0.20), (0.0, 0.0, -depth - 0.10))
    wall_height = depth + wall_top
    wall_centre = (wall_top - depth) / 2
    scene.wall_x0 = tray_box("WallX0", (0.10, width + 0.2, wall_height), (-length / 2 - 0.05, 0.0, wall_centre))
    scene.wall_x1 = tray_box("WallX1", (0.10, width + 0.2, wall_height), (length / 2 + 0.05, 0.0, wall_centre))
    scene.wall_y0 = tray_box("WallY0", (length, 0.10, wall_height), (0.0, -width / 2 - 0.05, wall_centre))
    scene.wall_y1 = tray_box("WallY1", (length, 0.10, wall_height), (0.0, width / 2 + 0.05, wall_centre))
    # Each solver owns its static shapes, so the rigid solver gets a hidden copy of every tray box.
    for name in ("floor", "wall_x0", "wall_x1", "wall_y0", "wall_y1"):
        mirror = deepcopy(getattr(scene, name))
        mirror.prim_path = mirror.prim_path.replace("}/", "}/Rigid")
        mirror.spawn.visible = False
        setattr(scene, "rigid_" + name, mirror)


def physics_cfg(bodies, proxies):
    """Implicit MPM sand and MuJoCo Warp rigid bodies, coupled by the lagged reaction of the sand on ``proxies``."""
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        use_mujoco_contacts=False,
                        integrator="implicitfast",
                        cone="pyramidal",
                        iterations=100,
                        nconmax=300,
                        njmax=1000,
                    ),
                    bodies=bodies,
                    include_static_shapes=False,
                    shape_label_patterns=[r"/World/envs/env_.*/Rigid(Floor|Wall[XY][01])/.*"],
                    substeps=2,
                ),
                CouplerEntryCfg(
                    name="sand",
                    solver_cfg=MPMSolverCfg(
                        voxel_size=VOXEL_SIZE,
                        grid_type="sparse",
                        grid_padding=0,
                        max_active_cell_count=1 << 18,
                        strain_basis="P0",
                        transfer_scheme="apic",
                        max_iterations=25,
                        tolerance=1e-4,
                        warmstart_mode="auto",
                        velocity_basis="Q1",
                        collider_basis="pic27",
                        collider_velocity_mode="forward",
                        solver="auto",
                        critical_fraction=0.0,
                        separate_worlds=True,
                        project_outside_colliders=False,
                    ),
                    all_particles=True,
                    include_static_shapes=False,
                    shape_label_patterns=[r"/World/envs/env_.*/(Floor|Wall[XY][01])/.*"],
                    in_place=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="sand",
                    bodies=proxies,
                    mode="lagged",
                    mass_scale=1.0,
                    collision_pipeline=None,
                )
            ],
            iterations=1,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        use_cuda_graph=False,
    )


class QuietObjectStateAction(ActionTerm):
    """Set the pose (xyz, quaternion xyzw) and world-frame velocity of a kinematic object.

    The pose and velocity are written with ``skip_forward=True``. The default write invalidates the cached data
    of the asset after every step, and with the MPM coupler that disturbs the sand: the whole bed keeps sinking
    (about 4 cm in 6 s) even when the object is parked far away. Nothing here reads the cached data back.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros((env.num_envs, 13), device=env.device)
        self._actions[:, 6] = 1

    @property
    def action_dim(self):
        return 13

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        self._actions.copy_(actions)

    def apply_actions(self):
        pose = self._actions[:, :7].clone()
        pose[:, :3] += self._env.scene.env_origins
        self._asset.write_root_pose_to_sim_index(root_pose=pose, skip_forward=True)
        self._asset.write_root_velocity_to_sim_index(root_velocity=self._actions[:, 7:].contiguous(), skip_forward=True)


class BodyWrenchAction(ActionTerm):
    """Apply a world-frame force and torque at the centre of mass of a free object."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros((env.num_envs, 6), device=env.device)

    @property
    def action_dim(self):
        return 6

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        self._actions.copy_(actions)

    def apply_actions(self):
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._actions[:, None, :3].contiguous(),
            torques=self._actions[:, None, 3:].contiguous(),
            is_global=True,
        )


def reset_sand(env, env_ids):
    """Put the particles back and clear the history of the MPM solver and of the coupling."""
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    sand = env.scene["sand"]
    sand.write_nodal_state_to_sim_index(sand.data.default_nodal_state_w.torch[ids].clone(), env_ids=ids)
    mask = torch.zeros(env.num_envs + 1, dtype=torch.bool, device=env.device)
    mask[ids] = True
    NewtonMPMManager.reset_solver_state(
        world_mask=None if len(ids) == env.num_envs else wp.from_torch(mask, dtype=wp.bool),
        flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
    )


def object_pose(env):
    """World pose of the object."""
    return env.scene["object"].data.root_pose_w.torch


def object_spawn_cfg(geometry: str):
    """Spawn settings of the rigid object: a ``box``, ``sphere`` or ``cylinder`` of 2 kg, kinematic by default."""
    common = dict(
        rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        mass_props=sim.MassCfg(mass=2.0),
        collision_props=NewtonCollisionCfg(contact_margin=0.0),
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.12, 0.32, 0.25)),
    )
    if geometry == "box":
        return sim.CuboidCfg(size=(0.24, 0.18, 0.12), **common)
    if geometry == "sphere":
        return sim.SphereCfg(radius=0.10, **common)
    if geometry == "cylinder":
        return sim.CylinderCfg(radius=0.10, height=0.12, axis="Z", **common)
    raise ValueError(f"Unknown object geometry: {geometry}")


@configclass
class SceneCfg(InteractiveSceneCfg):
    """One rigid object above a flat sand bed; :func:`make_env_cfg` sets the object shape and the terrain."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=object_spawn_cfg("box"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.65, 0.0, 0.30)),
    )

    def __post_init__(self):
        configure_terrain(self, "flat")


@configclass
class ActionsCfg:
    motion = ActionTermCfg(class_type=QuietObjectStateAction, asset_name="object")


@configclass
class ObservationsCfg:
    @configclass
    class StateCfg(ObservationGroupCfg):
        pose = ObservationTermCfg(func=object_pose)
        enable_corruption = False

    state = StateCfg()


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=reset_scene_to_default, mode="reset")
    reset_sand = EventTermCfg(func=reset_sand, mode="reset")


@configclass
class ObjectEnvCfg(ManagerBasedRLEnvCfg):
    """One environment, one action per physics step of 5 ms."""

    scene = SceneCfg(num_envs=1, env_spacing=4.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    decimation = 1
    seed = 42
    episode_length_s = 6.0
    rewards = {}
    terminations = {}
    sim = SimulationCfg(dt=0.005, render_interval=4, physics=physics_cfg(OBJECT_BODIES, OBJECT_BODIES))


def make_env_cfg(geometry: str, terrain: str, bed_size: tuple[float, float, float]) -> ObjectEnvCfg:
    """Environment of one object shape on one sand bed.

    Args:
        geometry: ``box``, ``sphere`` or ``cylinder``.
        terrain: Surface of the bed, as for :func:`surface_height`.
        bed_size: Length, width and depth of the bed [m].
    """
    cfg = ObjectEnvCfg()
    cfg.scene.object.spawn = object_spawn_cfg(geometry)
    configure_terrain(cfg.scene, terrain, bed_size)
    return cfg
