# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.utils import configclass


@configclass
class G1PhysicsCfg(NewtonCfg):
    """Newton MJWarp settings for the G1 whole-body task."""

    solver_cfg = MJWarpSolverCfg(
        njmax=1000,
        nconmax=300,
        cone="pyramidal",
        integrator="implicitfast",
        use_mujoco_contacts=False,
    )
    collision_cfg = NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
    default_shape_cfg = NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0)
    num_substeps = 2
