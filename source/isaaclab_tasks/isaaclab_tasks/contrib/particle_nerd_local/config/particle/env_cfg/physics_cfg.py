# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation and coupler configurations for the `particle` two-way coupling task."""

from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.sim import SimulationCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.contrib.particle_nerd_local import (
    DEFAULT_PARTICLE_NERD_LOCAL_CHECKPOINT,
    ParticleNeRDLocalSolverCfg,
)

from .scene_cfg import VOXEL_SIZE

ANCHOR_BODIES = [r"/World/envs/env_.*/Anchor"]


def make_particle_nerd_local_sim_cfg(checkpoint: str = DEFAULT_PARTICLE_NERD_LOCAL_CHECKPOINT) -> SimulationCfg:
    """Return coupled MuJoCo-Warp + `ParticleNeRDLocalSolverCfg` simulation config."""
    return SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                class_type="isaaclab_tasks.contrib.particle_nerd_local.manager:NewtonParticleNeRDLocalCouplerManager",
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
                        bodies=ANCHOR_BODIES,
                        include_static_shapes=False,
                        shape_label_patterns=[r"/World/envs/env_.*/Rigid(Floor|Wall[XY][01])/.*"],
                        substeps=2,
                    ),
                    CouplerEntryCfg(
                        name="sand",
                        solver_cfg=ParticleNeRDLocalSolverCfg(
                            checkpoint=checkpoint,
                            collider_body_patterns=[r".*/Anchor"],
                            bed_depth=0.30,
                            use_unified_sdf_colliders=False,
                            effective_mass_scale=2.0,
                            tangent_mass_scale=2.0,
                            horizontal_mass_scale=2.0,
                            upward_mass_scale=2.0,
                            contact_relaxation=0.12,
                            contact_iterations=200,
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
                        bodies=ANCHOR_BODIES,
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
        ),
    )


def make_mpm_coupler_sim_cfg() -> SimulationCfg:
    """Return ground-truth coupled MuJoCo-Warp + Implicit MPM simulation config."""
    return SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=NewtonCfg(
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
                        bodies=ANCHOR_BODIES,
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
                        bodies=ANCHOR_BODIES,
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
        ),
    )
