# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.contrib.particle_nerd import ParticleNeRDSolverCfg

RIGID_ENTRY = "robot"
"""Name of the MJWarp rigid entry."""
PARTICLE_ENTRY = "sand"
"""Name of the Particle NeRD entry."""
RIGID_PROXY_BODIES = [r"/World/envs/env_[^/]+/.*"]
"""Rigid bodies integrated by MJWarp."""
FOOT_PROXY_BODIES = [r"/World/envs/env_.*/Robot/.*ankle_roll_link"]
"""Foot bodies passed to the neural solver as colliders."""
DEFAULT_PROXY_MASS_SCALE = 1.0
"""Dimensionless effective-mass multiplier for rigid proxies."""


@configclass
class G1ParticleNeRDPhysicsCfg(NewtonCfg):
    """Two-way MJWarp/Particle NeRD coupling with fixed-size particle recycling."""

    solver_cfg: CouplerProxyCfg = CouplerProxyCfg(
        class_type="isaaclab_tasks.contrib.particle_nerd.manager:NewtonParticleNeRDCouplerManager",
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
                bodies=RIGID_PROXY_BODIES,
                shape_label_patterns=[r"/World/ground/terrain/.*"],
                include_static_shapes=False,
                # Refine the articulation integration between coupled exchanges.
                substeps=2,
            ),
            CouplerEntryCfg(
                name=PARTICLE_ENTRY,
                solver_cfg=ParticleNeRDSolverCfg(
                    checkpoint=str(Path(__file__).resolve().parents[2] / "particle_nerd/checkpoint/best.pt"),
                    # Match the validated fixed-terrain task; retain moving-patch coordinates.
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
                destination=PARTICLE_ENTRY,
                bodies=FOOT_PROXY_BODIES,
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
    capture_solver: bool = True
    """Capture GPU steps with PyTorch-owned allocation pools; particle recycling remains eager."""

    def configure_terrain(self, terrain, proxy_mass_scale):
        for mapping in self.solver_cfg.proxies:
            mapping.mass_scale = proxy_mass_scale
        self.solver_cfg.entries[1].solver_cfg.bed_depth = terrain.particle_depth
