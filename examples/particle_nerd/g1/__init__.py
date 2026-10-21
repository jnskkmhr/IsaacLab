# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 material-motion collection, Transolver training and evaluation."""

from pathlib import Path

DEFAULT_POLICY = (
    Path.home() / "isaac/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29dof_mpm"
    / "benchmark/data/checkpoints/5f1xyc21/model_39998.pt"
)
MATERIAL_KEYS = (
    "density", "young_modulus", "poisson_ratio", "friction", "yield_pressure",
    "tensile_yield_ratio", "yield_stress", "hardening", "dilatancy", "viscosity",
)
