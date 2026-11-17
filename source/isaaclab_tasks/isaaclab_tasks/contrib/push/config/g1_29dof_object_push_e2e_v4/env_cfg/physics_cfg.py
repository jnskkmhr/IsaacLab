# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local Newton/MJWarp settings for rigid-ground pushing."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.utils import configclass


@configclass
class G1PhysicsCfg(NewtonCfg):
    """Simulate the robot, box, and ground in one rigid-body solver."""

    solver_cfg = MJWarpSolverCfg(
        use_mujoco_contacts=False,
        integrator="implicitfast",
        cone="pyramidal",
        impratio=1.0,
        nconmax=300,
        njmax=1000,
        iterations=100,
    )
    collision_cfg = NewtonCollisionPipelineCfg(soft_contact_max=0)
    default_shape_cfg = NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0)
    num_substeps = 2
    use_cuda_graph = True
