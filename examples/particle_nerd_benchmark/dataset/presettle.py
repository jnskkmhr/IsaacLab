# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sand-only implicit MPM outside the simulation app, used only to pre-settle the bed of a robot episode.

Object and sand-only episodes do not use this: they settle and run in the coupled scene of `core.physics`, so their
solver settings are identical. The robot collector still settles its bed here before the robot is loaded, because
the robot cannot be parked away from the bed. The solver settings below differ from the coupled scene (collider
basis, two half steps per frame), so do not use this for anything that is recorded.
"""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from ..core.terrain import DENSITY, SPACING, VOXEL_SIZE, sand_cfg
from .episodes import DT

MATERIAL = sand_cfg().spawn.material.to_dict()
"""Sand material of every episode (the one defined in `core.terrain.sand_cfg`)."""


def simulate(
    points: np.ndarray, velocity: np.ndarray, boxes: list[dict], steps: int, device: str, settle_steps: int = 0
) -> np.ndarray:
    """Run implicit MPM at 200 Hz (two half steps per frame) and return particle positions [steps + 1, N, 3].

    ``settle_steps`` are simulated first and not recorded: frame 0 is the state after settling.
    """
    n = len(points)
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
    SolverImplicitMPM.register_custom_attributes(builder)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0, margin=0.0)
    for box in boxes:
        half = np.asarray(box["size_m"], dtype=np.float32) / 2.0
        pose = box["pose"]
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])),
            hx=float(half[0]),
            hy=float(half[1]),
            hz=float(half[2]),
            cfg=shape_cfg,
        )
    builder.add_particles(
        pos=points.tolist(),
        vel=velocity.tolist(),
        mass=[float(DENSITY * SPACING**3)] * n,
        radius=[SPACING / 2.0] * n,
    )
    model = builder.finalize(device=device)
    model.particle_mu = MATERIAL["friction"]
    for name in ("young_modulus", "poisson_ratio", "yield_pressure", "yield_stress", "tensile_yield_ratio"):
        getattr(model.mpm, name).fill_(MATERIAL[name])
    model.mpm.hardening.fill_(MATERIAL["hardening"])
    model.mpm.dilatancy.fill_(MATERIAL["dilatancy"])
    model.mpm.friction.fill_(MATERIAL["friction"])
    config = SolverImplicitMPM.Config()
    config.voxel_size = VOXEL_SIZE
    config.grid_type = "sparse"
    config.max_active_cell_count = 1 << 18
    config.strain_basis = "P0"
    config.velocity_basis = "Q1"
    config.transfer_scheme = "apic"
    config.max_iterations = 25
    config.tolerance = 1.0e-4
    config.critical_fraction = 0.0
    config.collider_velocity_mode = "forward"
    solver = SolverImplicitMPM(model, config)
    s0, s1 = model.state(), model.state()
    for _ in range(settle_steps):
        for _ in range(2):
            solver.step(s0, s1, None, None, DT / 2.0)
            s0, s1 = s1, s0
    out = np.empty((steps + 1, n, 3), dtype=np.float32)
    out[0] = s0.particle_q.numpy()
    for step in range(steps):
        for _ in range(2):
            solver.step(s0, s1, None, None, DT / 2.0)
            s0, s1 = s1, s0
        out[step + 1] = s0.particle_q.numpy()
    return out
