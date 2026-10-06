# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for custom MJWarp coupling with VBD or MPM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonSolverCfg, VBDSolverCfg

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the custom MJWarp and VBD coupling manager."""

    class_type: type[NewtonManager] | str = "{DIR}.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    """Manager class for the coupled solver."""

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MJWarp rigid-body solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD deformable solver configuration."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Coupling direction between the rigid and deformable solvers."""


@configclass
class CoupledMJWarpMPMSolverCfg(CouplerCfg):
    """Direct lagged MPM wrench feedback, following Newton's two-way MPM example.

    The MPM entry advances once per coupled timestep. The MJWarp entry can substep
    while holding the previous MPM wrench constant over that interval.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_mjwarp_mpm_manager:NewtonCoupledMJWarpMPMManager"
    """Manager class for direct MJWarp/MPM coupling."""

    rigid_entry: str = "robot"
    """Name of the MJWarp solver entry."""

    mpm_entry: str = "sand"
    """Name of the implicit-MPM solver entry."""

    collider_bodies: list[str | int] = []
    """Parent-model bodies presented to MPM as rigid colliders."""
