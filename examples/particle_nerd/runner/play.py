# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run current GNN particles, analytical material force, and a dynamic box; no MPM."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from particle_nerd.data_collection.dataset import initialize_predictor
from particle_nerd.runner.simulation import CoupledSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True, help="Only the settled initial state/history is read")
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional diagnostic JSON log")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    predictor, initial = initialize_predictor(args.checkpoint, args.episode, args.selector, args.compile)
    simulation = CoupledSimulation(
        predictor,
        initial["manifest"],
        initial["particle_mass"],
        initial["body_pose"],
        initial["body_velocity"],
        max_iterations=args.max_iterations,
    )
    viewer = None
    if not args.headless:
        from newton.viewer import ViewerGL

        viewer = ViewerGL(width=1200, height=800)
        viewer.set_model(simulation.rigid.model)
        viewer.set_camera(wp.vec3(1.8, -2.4, 1.8))
        viewer.camera.look_at((0.0, 0.0, 0.18))
        colors = wp.array(
            np.tile([0.32, 0.24, 0.21], (len(predictor.x), 1)).astype("f4"), dtype=wp.vec3, device=predictor.device
        )
        mouse_forces = simulation.rigid.model.state()
    print(
        "Experimental coupled runner: GNN + analytical force + MJWarp; no MPM or recorded future body motion",
        flush=True,
    )
    rows = []
    started = time.perf_counter()
    try:
        for step in range(args.steps):
            if viewer is not None:
                if not viewer.is_running():
                    break
                with wp.ScopedStream(simulation.stream):
                    wp.copy(mouse_forces.body_q, simulation.rigid.state.body_q)
                    wp.copy(mouse_forces.body_qd, simulation.rigid.state.body_qd)
                    mouse_forces.clear_forces()
                    viewer.apply_forces(mouse_forces)
                simulation.step(mouse_forces.body_f)
            else:
                simulation.step()
            report = simulation.diagnostics()
            if viewer is not None:
                viewer.begin_frame((step + 1) * predictor.dt)
                viewer.log_state(simulation.rigid.state)
                viewer.log_points("sand", wp.from_torch(predictor.x, dtype=wp.vec3), radii=0.012, colors=colors)
                viewer.end_frame()
                remaining = (step + 1) * predictor.dt - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
            rows.append(report)
            if step % 25 == 0 or step == args.steps - 1:
                print(json.dumps(report), flush=True)
            if (
                report["particle"]["graph_overflow"]
                or not report["particle"]["finite"]
                or not report["contact"]["finite"]
            ):
                raise RuntimeError("Invalid coupled state; inspect the diagnostic log")
    finally:
        if viewer is not None:
            viewer.close()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(mode="experimental-coupled", steps=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
