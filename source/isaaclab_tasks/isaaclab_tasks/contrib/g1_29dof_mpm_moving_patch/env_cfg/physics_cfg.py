# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone coupled MJWarp/MPM configuration."""

from __future__ import annotations

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from .scene_cfg import MovingPatchTerrainCfg

RIGID_ENTRY = "robot"
"""Name of the MJWarp coupler entry."""

MPM_ENTRY = "sand"
"""Name of the implicit-MPM coupler entry."""

FOOT_PROXY_BODIES = [r"/World/envs/env_.*/Robot/.*ankle_roll_link"]
"""Rigid bodies handed to the MPM solver as colliders."""

DEFAULT_PROXY_MASS_SCALE = 1.0
"""Effective-mass scale applied to the proxied feet.

Mirrors ``coupling_relaxation`` of the standalone Newton G1 sand example: the G1 weighs about
32.3 kg while a single ankle roll link weighs 0.608 kg, so the two feet must present the whole
body mass to the granular solver for the robot to be supported rather than sink.
"""


@configclass
class G1PhysicsCfg(NewtonCfg):
    """Newton two-way rigid/MPM coupling for the moving-patch task."""

    solver_cfg: CouplerProxyCfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name=RIGID_ENTRY,
                solver_cfg=MJWarpSolverCfg(
                    use_mujoco_contacts=False,
                    integrator="implicitfast",
                    cone="pyramidal",
                    impratio=1.0,
                    # Falls can create hundreds of rigid contacts. Pyramidal friction
                    # expands each contact into multiple constraint rows.
                    nconmax=300,
                    njmax=1000,
                    iterations=100,
                ),
                bodies=[r"/World/envs/env_.*/Robot"],
                shape_label_patterns=[r"/World/ground/terrain/.*"],
                include_static_shapes=False,
                # Refine the articulation integration between coupled exchanges.
                substeps=3,
            ),
            CouplerEntryCfg(
                name=MPM_ENTRY,
                solver_cfg=MPMSolverCfg(
                    check_particle_grid_mapping=True,
                    voxel_size=MovingPatchTerrainCfg().voxel_size,
                    grid_type="sparse",
                    # A sparse grid stays rebuildable, and therefore CUDA-graph capturable,
                    # only while it is unpadded. The standalone Newton examples pad by 50
                    # voxels, but only on a fixed grid; padding a sparse grid here costs an
                    # order of magnitude in step time and overruns the node capacity.
                    grid_padding=0,
                    strain_basis="P0",
                    transfer_scheme="apic",
                    max_iterations=25,
                    # tolerance=1.0e-5,
                    # warmstart_mode="auto",
                    # velocity_basis="Q1",
                    # collider_basis="S2",
                    tolerance=1.0e-4,
                    warmstart_mode="auto",
                    velocity_basis="Q1",
                    collider_basis="pic27",
                    collider_velocity_mode="forward",
                    solver="auto",
                    # Voxel fill fraction below which the yield surface collapses. The bed is
                    # sampled at a spacing that does not divide the voxel size, so its cells
                    # straddle the 0.5 the standalone G1 example uses and half the bed loses
                    # its shear strength; keep the Newton default so the bed stays granular.
                    critical_fraction=0.0,
                    separate_worlds=True,
                    project_outside_colliders=False,
                ),
                bodies=[],
                shape_label_patterns=[r"/World/ground/mpm_support/.*"],
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
    use_cuda_graph: bool = False

    def configure_terrain(self, moving_patch_terrain: MovingPatchTerrainCfg, proxy_mass_scale: float) -> None:
        """Apply final terrain settings without replacing user-configured solvers."""
        if not isinstance(self.solver_cfg, CouplerProxyCfg):
            raise ValueError("This task requires CouplerProxyCfg")
        for proxy in self.solver_cfg.proxies:
            proxy.mass_scale = proxy_mass_scale
        for entry in self.solver_cfg.entries:
            if entry.name == MPM_ENTRY:
                entry.solver_cfg.voxel_size = moving_patch_terrain.voxel_size
