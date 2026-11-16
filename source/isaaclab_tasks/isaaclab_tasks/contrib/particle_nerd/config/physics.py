# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replace WorldParticle inference while retaining the reference coupling settings."""

from copy import deepcopy
from dataclasses import fields

from ..object_solver_cfg import ParticleNeRDObjectSolverCfg


def use_object_nerd(physics):
    """Copy a reference physics config and replace its sand solver with object NeRD."""
    settings = {
        field.name: deepcopy(getattr(physics, field.name)) for field in fields(physics) if field.name != "class_type"
    }
    coupler = settings["solver_cfg"]
    coupler.class_type = "isaaclab_tasks.contrib.particle_nerd.object_manager:NewtonParticleNeRDObjectCouplerManager"
    entry = next(entry for entry in coupler.entries if entry.name == "sand")
    entry.solver_cfg = ParticleNeRDObjectSolverCfg()
    return type(physics)(**settings)
