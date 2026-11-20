# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled MJWarp/MPM physics presets: which particles are active and whether the sand streams.

Select a preset with ``physics=NAME``, for example ``physics=solver_cfg_strain_activation``.
"""

from __future__ import annotations

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSleepingCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    ParticleCacheCfg,
    StrainActivationCfg,
    StreamingCfg,
    TrackedRegionActivationCfg,
)
from isaaclab_newton.physics.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from ....util.solver_setting import MPM_ENTRY
from .streaming_scene_cfg import VOXEL_SIZE, SandLayerCfg

RIGID_ENTRY = "robot"
"""Name of the MJWarp coupler entry."""

FOOT_PROXY_BODIES = [r"/World/envs/env_.*/Robot/.*ankle_roll_link"]
"""Rigid bodies handed to the MPM solver as colliders."""

DEFAULT_PROXY_MASS_SCALE = 1.0
"""Effective-mass scale applied to the proxied feet.
"""

# MAX_CACHED_PARTICLES_PER_ENV = 500_000
# MAX_CACHED_PARTICLES_PER_ENV = 1000_000
MAX_CACHED_PARTICLES_PER_ENV = 1500_000
"""Maximum number of cached particles per environment."""

"""
Fixed sand bed: every spawned particle stays in place.
"""


@configclass
class G1PhysicsProxyCfgAllParticleActive(NewtonCfg):
    """Newton two-way rigid/MPM coupling with a fixed, walled sand bed; every particle is simulated."""

    solver_cfg: CouplerProxyCfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name=RIGID_ENTRY,
                solver_cfg=MJWarpSolverCfg(
                    use_mujoco_contacts=False,
                    integrator="implicitfast",
                    cone="pyramidal",
                    impratio=1.0,
                    nconmax=300,
                    njmax=1000,
                    iterations=100,
                ),
                bodies=[r"/World/envs/env_.*/Robot"],
                # picks up the hidden static pan floor, which has no rigid body to name
                include_static_shapes=True,
                # Refine the articulation integration between coupled exchanges.
                substeps=2,
            ),
            CouplerEntryCfg(
                name=MPM_ENTRY,
                solver_cfg=MPMSolverCfg(
                    voxel_size=VOXEL_SIZE,
                    grid_type="sparse",
                    grid_padding=0,
                    strain_basis="P0",
                    transfer_scheme="apic",
                    max_iterations=25,
                    tolerance=1.0e-4,
                    warmstart_mode="auto",
                    velocity_basis="Q1",
                    collider_basis="pic27",
                    collider_velocity_mode="forward",
                    solver="auto",
                    critical_fraction=0.0,
                    separate_worlds=True,
                    project_outside_colliders=False,
                ),
                # Retaining boxes of the flat bed, or the support body of a generated sand surface.
                bodies=[r"/World/envs/env_.*/(Sand/)?MPM(BedFloor|BedWallFront|BedWallBack|BedWallLeft|BedWallRight)"],
                all_particles=True,
                include_static_shapes=False,
                include_child_joints=False,
                substeps=1,
                in_place=True,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source=RIGID_ENTRY,
                destination=MPM_ENTRY,
                bodies=FOOT_PROXY_BODIES,  # type: ignore
                mode="lagged",
                mass_scale=DEFAULT_PROXY_MASS_SCALE,
                collision_pipeline=None,
            )
        ],
        iterations=1,
    )

    collision_cfg: NewtonCollisionPipelineCfg = NewtonCollisionPipelineCfg(soft_contact_max=0)
    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0)
    num_substeps: int = 1
    use_cuda_graph: bool = True


@configclass
class G1PhysicsProxyCfgStrainActive(G1PhysicsProxyCfgAllParticleActive):
    """Fixed sand bed in which settled sand sleeps until it is strained again."""

    def __post_init__(self):
        mpm = self.solver_cfg.entries[1].solver_cfg
        mpm.reuse_grid_topology = True
        mpm.sleeping = MPMSleepingCfg(strain=StrainActivationCfg(stress_filter_time=0.1))


@configclass
class G1PhysicsProxyCfgRegionActive(G1PhysicsProxyCfgAllParticleActive):
    """Fixed sand bed in which only the sand in a box around the pelvis is simulated."""

    def __post_init__(self):
        mpm = self.solver_cfg.entries[1].solver_cfg
        mpm.reuse_grid_topology = True
        mpm.sleeping = MPMSleepingCfg(
            strain=None,
            region=TrackedRegionActivationCfg(body_names="pelvis", size=(1.2, 1.2, 3.0), keep_material_motion=False),
        )


@configclass
class G1PhysicsProxyCfg(PresetCfg):
    """Activation presets for a fixed sand bed."""

    solver_cfg_all_particle = G1PhysicsProxyCfgAllParticleActive()
    solver_cfg_strain_activation = G1PhysicsProxyCfgStrainActive()
    solver_cfg_region_activation = G1PhysicsProxyCfgRegionActive()
    default = solver_cfg_all_particle


"""
Streaming sand: a pool of particles follows the robot over the shared terrain.
"""


@configclass
class G1StreamPhysicsProxyCfgAllParticleActiveNoCache(NewtonCfg):
    """Newton two-way rigid/MPM coupling with a streaming sand pool under the robot."""

    solver_cfg: CouplerProxyCfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name=RIGID_ENTRY,
                solver_cfg=MJWarpSolverCfg(
                    use_mujoco_contacts=False,
                    integrator="implicitfast",
                    cone="pyramidal",
                    impratio=1.0,
                    nconmax=300,
                    njmax=1000,
                    iterations=100,
                ),
                bodies=[r"/World/envs/env_.*/Robot"],
                shape_label_patterns=[r"/World/ground/terrain/.*"],
                include_static_shapes=False,
                # Refine the articulation integration between coupled exchanges.
                substeps=2,
            ),
            CouplerEntryCfg(
                name=MPM_ENTRY,
                solver_cfg=MPMSolverCfg(
                    voxel_size=VOXEL_SIZE,
                    grid_type="sparse",
                    grid_padding=0,
                    strain_basis="P0",
                    transfer_scheme="apic",
                    max_iterations=25,
                    tolerance=1.0e-4,
                    warmstart_mode="auto",
                    velocity_basis="Q1",
                    collider_basis="pic27",
                    collider_velocity_mode="forward",
                    solver="auto",
                    critical_fraction=0.0,
                    separate_worlds=True,
                    project_outside_colliders=False,
                    # Fixed pool of sand that follows the pelvis; the cache is not used.
                    streaming=StreamingCfg(body_names="pelvis", size=(1.2, 1.2), tile_size=0.2, margin=0.2),
                ),
                bodies=[],
                shape_label_patterns=[r"/World/ground/mpm_ground/.*"],
                all_particles=True,
                include_static_shapes=False,
                include_child_joints=False,
                substeps=1,
                in_place=True,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source=RIGID_ENTRY,
                destination=MPM_ENTRY,
                bodies=FOOT_PROXY_BODIES,  # type: ignore
                mode="lagged",
                mass_scale=DEFAULT_PROXY_MASS_SCALE,
                collision_pipeline=None,
            )
        ],
        iterations=1,
    )

    collision_cfg: NewtonCollisionPipelineCfg = NewtonCollisionPipelineCfg(soft_contact_max=0)
    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0)
    num_substeps: int = 1
    use_cuda_graph: bool = True

    def configure_sand(self, sand: SandLayerCfg, proxy_mass_scale: float) -> None:
        """Apply final sand settings without replacing user-configured solvers."""
        for proxy in self.solver_cfg.proxies:
            proxy.mass_scale = proxy_mass_scale
        for entry in self.solver_cfg.entries:
            if entry.name == MPM_ENTRY:
                entry.solver_cfg.voxel_size = sand.voxel_size  # type: ignore


@configclass
class G1StreamPhysicsProxyCfgAllParticleActiveCache(G1StreamPhysicsProxyCfgAllParticleActiveNoCache):
    """Streaming sand pool with every particle simulated; a tile cache remembers the sand left behind."""

    def __post_init__(self):
        self.solver_cfg.entries[1].solver_cfg.streaming.cache = ParticleCacheCfg(
            max_cached_particles_per_env=MAX_CACHED_PARTICLES_PER_ENV
        )


@configclass
class G1StreamPhysicsProxyCfgNoCache(G1StreamPhysicsProxyCfgAllParticleActiveNoCache):
    """Streaming sand pool in which settled sand sleeps; sand left behind is not remembered.

    Sand stays awake only inside a box around the pelvis, and settled sand inside that box sleeps too.
    """

    def __post_init__(self):
        mpm = self.solver_cfg.entries[1].solver_cfg
        mpm.reuse_grid_topology = True
        mpm.sleeping = MPMSleepingCfg(
            strain=StrainActivationCfg(stress_filter_time=0.1),
            region=TrackedRegionActivationCfg(body_names="pelvis", size=(1.2, 1.2, 3.0)),
        )


@configclass
class G1StreamPhysicsProxyCfgCache(G1StreamPhysicsProxyCfgNoCache):
    """Streaming sand pool in which settled sand sleeps; a tile cache remembers the sand left behind."""

    def __post_init__(self):
        super().__post_init__()
        self.solver_cfg.entries[1].solver_cfg.streaming.cache = ParticleCacheCfg(
            max_cached_particles_per_env=MAX_CACHED_PARTICLES_PER_ENV
        )


@configclass
class G1StreamPhysicsProxyCfg(PresetCfg):
    """Activation and caching presets for streaming sand."""

    solver_cfg_all_active_no_cache = G1StreamPhysicsProxyCfgAllParticleActiveNoCache()
    solver_cfg_all_active_cache = G1StreamPhysicsProxyCfgAllParticleActiveCache()
    solver_cfg_no_cache = G1StreamPhysicsProxyCfgNoCache()
    solver_cfg_cache = G1StreamPhysicsProxyCfgCache()
    default = solver_cfg_cache
