# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton rigid-body solver coupled to the reference analytical RFT foot forces."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.utils import configclass


@configclass
class G1PhysicsCfg(NewtonCfg):
    solver_cfg = MJWarpSolverCfg(
        njmax=1000,
        nconmax=300,
        cone="pyramidal",
        impratio=1.0,
        integrator="implicitfast",
        use_mujoco_contacts=False,
    )
    collision_cfg = NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
    num_substeps = 2
    default_shape_cfg = NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0)
