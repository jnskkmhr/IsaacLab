# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled MJWarp/MPM physics presets for the tow task: the robot's feet and the sled press on the sand.

Select a preset with ``physics=NAME``, for example ``physics=solver_cfg_all_active_cache``.
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

from isaaclab_tasks.contrib.velocity_mpm.util.solver_setting import MPM_ENTRY
from isaaclab_tasks.utils import PresetCfg

from .scene_cfg import SLED_NAME, VOXEL_SIZE, SandLayerCfg

RIGID_ENTRY = "robot"
"""Name of the MJWarp coupler entry."""

SLED_BODY = r"/World/envs/env_.*/" + SLED_NAME
"""The sled as a rigid body of the MJWarp entry and as a collider of the MPM entry."""

PROXY_BODIES = [r"/World/envs/env_.*/Robot/.*ankle_roll_link", SLED_BODY]
"""Rigid bodies handed to the MPM solver as colliders."""

ANCHOR_BODY_NAMES = ["pelvis", SLED_NAME]
"""Bodies that each carry a streaming pool, and around which the sleeping preset keeps the sand awake."""

DEFAULT_PROXY_MASS_SCALE = 1.0
"""Effective-mass scale applied to the proxied bodies."""

MAX_CACHED_PARTICLES_PER_ENV = 1500_000
"""Maximum number of cached particles per environment."""


@configclass
class G1TowPhysicsCfgAllParticleActive(NewtonCfg):
    """Newton two-way rigid/MPM coupling; the robot and the sled each carry a cached streaming pool."""

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
                bodies=[r"/World/envs/env_.*/Robot", SLED_BODY],
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
                    # One pool per anchor body. Several pools need the tile cache, which also keeps the trench
                    # the sled leaves behind.
                    streaming=StreamingCfg(
                        body_names=ANCHOR_BODY_NAMES,
                        size=(1.2, 1.2),
                        tile_size=0.2,
                        margin=0.2,
                        # Two pools that do not overlap load twice the tiles of one; the rest is displaced sand.
                        cache=ParticleCacheCfg(
                            max_cached_particles_per_env=MAX_CACHED_PARTICLES_PER_ENV,
                            simulation_capacity_multiplier=4.0,
                        ),
                    ),
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
                bodies=PROXY_BODIES,  # type: ignore
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
class G1TowPhysicsCfgCache(G1TowPhysicsCfgAllParticleActive):
    """Cached streaming pools in which settled sand sleeps.

    Sand stays awake only inside a box around the pelvis and the sled, and settled sand inside those boxes
    sleeps too.
    """

    def __post_init__(self):
        mpm = self.solver_cfg.entries[1].solver_cfg
        mpm.reuse_grid_topology = True
        mpm.sleeping = MPMSleepingCfg(
            strain=StrainActivationCfg(stress_filter_time=0.1),
            region=TrackedRegionActivationCfg(body_names=list(ANCHOR_BODY_NAMES), size=(1.2, 1.2, 3.0)),
        )


@configclass
class G1TowPhysicsCfg(PresetCfg):
    """Activation presets for the cached streaming pools, named as in the walking task."""

    solver_cfg_all_active_cache = G1TowPhysicsCfgAllParticleActive()
    solver_cfg_cache = G1TowPhysicsCfgCache()
    default = solver_cfg_all_active_cache
