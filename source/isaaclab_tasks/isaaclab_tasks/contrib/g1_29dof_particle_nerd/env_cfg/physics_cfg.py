# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled MJWarp + Particle NeRD physics for the G1 granular locomotion task.

The rigid entry integrates the articulation with MJWarp, the neural entry integrates the granular
bed, and a lagged proxy mapping hands the feet to the neural solver as colliders. Only the ankle
roll links are proxied: they are the sole bodies expected to touch the bed, and every proxied
body contributes a collider to the neural features.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.contrib.particle_nerd.solver_cfg import ParticleNeRDSolverCfg

RIGID_ENTRY = "robot"
"""Name of the MJWarp coupler entry."""

PARTICLE_ENTRY = "sand"
"""Name of the particle coupler entry, matching the fixed MPM reference."""

FOOT_PROXY_BODIES = [r"/World/envs/env_.*/Robot/.*ankle_roll_link"]
"""Rigid bodies handed to the neural solver as colliders."""

DEFAULT_PROXY_MASS_SCALE = 1.0
"""Scale applied to the foot proxy masses in the neural solver."""


@configclass
class G1ParticleNeRDPhysicsCfg(NewtonCfg):
    """Coupled physics with PyTorch-owned CUDA graph capture for neural inference."""

    capture_solver: bool = True


def g1_particle_nerd_physics_cfg(
    proxy_mass_scale: float = DEFAULT_PROXY_MASS_SCALE, *, rough_terrain: bool = False
) -> NewtonCfg:
    """Build the coupled MJWarp/Particle NeRD physics configuration.

    Args:
        proxy_mass_scale: Scale of the virtual foot inertia used by contact reconstruction.
        rough_terrain: Select the terrain spawner's MPM support body instead of flat retaining boxes.

    Returns:
        The Newton physics configuration.
    """
    return G1ParticleNeRDPhysicsCfg(
        solver_cfg=CouplerProxyCfg(
            class_type="isaaclab_tasks.contrib.particle_nerd.manager:NewtonParticleNeRDCouplerManager",
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
                    name=PARTICLE_ENTRY,
                    solver_cfg=ParticleNeRDSolverCfg(
                        checkpoint=str(Path(__file__).resolve().parents[2] / "particle_nerd/checkpoint/best.pt"),
                        # Pin the recorded demo settings instead of inheriting experimental defaults.
                        active_k="auto",
                        selection_interval=4,
                        anchor_particles=512,
                        effective_mass_scale=100.0,
                        tangent_mass_scale=40.0,
                        horizontal_mass_scale=8.0,
                        upward_mass_scale=2.0,
                        normal_power=3.0,
                        contact_iterations=200,
                        contact_relaxation=0.02,
                        gradient_radius=0.065,
                        gradient_contact_band=0.08,
                        contact_stabilization=0.01,
                        contact_offset=0.0,
                    ),
                    bodies=[r"/World/envs/env_.*/Sand/MPMBedFloor"]
                    if rough_terrain
                    else [
                        r"/World/envs/env_.*/MPMBedFloor",
                        r"/World/envs/env_.*/MPMBedWallFront",
                        r"/World/envs/env_.*/MPMApproachBank",
                        r"/World/envs/env_.*/MPMBedWallLeft",
                        r"/World/envs/env_.*/MPMBedWallRight",
                    ],
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
                    destination=PARTICLE_ENTRY,
                    bodies=FOOT_PROXY_BODIES,  # type: ignore
                    mode="lagged",
                    mass_scale=proxy_mass_scale,
                    collision_pipeline=None,
                )
            ],
            iterations=1,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        # Newton's default shape stiffness (ke 2.5e3, kd 1e2) is two orders of magnitude too
        # compliant for a 32 kg humanoid: `MJWarpSolverCfg.use_mujoco_contacts=False` routes it
        # through `convert_solref`, so the rigid approach platform answers a footfall like a
        # mattress. These are the values the rigid and soft-contact G1 tasks walk on.
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        num_substeps=1,
        use_cuda_graph=False,
        capture_solver=True,
    )


def configure_fixed_bed(cfg) -> None:
    """Match neural bed coordinates to the final particle bounds and terrain geometry."""
    spawn = cfg.scene.sand.spawn
    solver = next(entry.solver_cfg for entry in cfg.sim.physics.solver_cfg.entries if entry.name == PARTICLE_ENTRY)
    bed = cfg.fixed_bed
    bed.lower, bed.upper = tuple(spawn.lower), tuple(spawn.upper)
    solver.bed_depth = bed.upper[2] - bed.lower[2]
    if cfg.scene.testbed is not None:
        bed.kind = "tilting"
    elif getattr(spawn, "terrain_generator", None) is not None:
        terrain = next(iter(spawn.terrain_generator.sub_terrains.values()))
        if hasattr(terrain, "slope_range"):
            if terrain.slope_range[0] != terrain.slope_range[1]:
                raise ValueError("Fixed NeRD diagnostics require a single specified slope")
            bed.kind, bed.slope = "slope", float(terrain.slope_range[0])
        else:
            if terrain.height_range[0] != terrain.height_range[1]:
                raise ValueError("Fixed NeRD diagnostics require a single specified wave height")
            bed.kind, bed.wave_height, bed.wave_count = "wave", float(terrain.height_range[0]), terrain.num_waves
    else:
        bed.kind = "flat"
    if cfg.events.mpm_material is not None or cfg.events.mpm_material_log is not None:
        raise ValueError("NeRD material features are fixed at construction; disable material reset randomization")
    if cfg.sim.physics.use_cuda_graph:
        raise ValueError("Use capture_solver=True and use_cuda_graph=False for Particle NeRD")
