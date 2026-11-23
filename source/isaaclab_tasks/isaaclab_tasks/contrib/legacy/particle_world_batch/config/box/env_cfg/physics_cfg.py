# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics configuration for the `box` two-way coupling task (`WorldParticle` and `Implicit MPM`)."""

from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.contrib.legacy.particle_world import ParticleWorldSolverCfg
from isaaclab_tasks.contrib.legacy.particle_world.solver_cfg import DEFAULT_WORLD_PARTICLE_CHECKPOINT

from .scene_cfg import BOX_VOXEL_SIZE

BOX_BODIES = [r"/World/envs/env_.*/Object"]
DEFAULT_CHECKPOINT = DEFAULT_WORLD_PARTICLE_CHECKPOINT


def box_particle_world_physics_cfg(checkpoint: str = DEFAULT_CHECKPOINT) -> NewtonCfg:
    """Two-way lagged coupling between MJWarp rigid bodies and `ParticleWorldSolver`."""
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            class_type="isaaclab_tasks.contrib.legacy.particle_world.manager:NewtonParticleWorldCouplerManager",
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
                    bodies=BOX_BODIES,
                    include_static_shapes=False,
                    shape_label_patterns=[r"/World/envs/env_.*/Rigid(Floor|Wall[XY][01])/.*"],
                    substeps=4,
                ),
                CouplerEntryCfg(
                    name="sand",
                    solver_cfg=ParticleWorldSolverCfg(
                        checkpoint=checkpoint,
                        collider_body_patterns=[r".*/Object"],
                        effective_mass_scale=10.0,
                        tangent_mass_scale=10.0,
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
                    bodies=BOX_BODIES,
                    mode="lagged",
                    mass_scale=1.0,
                    collision_pipeline=None,
                )
            ],
            iterations=1,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0, mu=0.5),
        use_cuda_graph=False,
    )


def box_mpm_physics_cfg() -> NewtonCfg:
    """Reference two-way coupling with Newton `SolverImplicitMPM` matching `example_mpm_twoway_coupling.py`."""
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
                    bodies=BOX_BODIES,
                    include_static_shapes=False,
                    shape_label_patterns=[r"/World/envs/env_.*/Rigid(Floor|Wall[XY][01])/.*"],
                    substeps=4,
                ),
                CouplerEntryCfg(
                    name="sand",
                    solver_cfg=MPMSolverCfg(
                        voxel_size=BOX_VOXEL_SIZE,
                        grid_type="sparse",
                        grid_padding=0,
                        max_active_cell_count=1 << 18,
                        strain_basis="P0",
                        transfer_scheme="apic",
                        max_iterations=50,
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
                    bodies=BOX_BODIES,
                    mode="lagged",
                    mass_scale=1.0,
                    collision_pipeline=None,
                )
            ],
            iterations=1,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0, mu=0.5),
        use_cuda_graph=False,
    )
