# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Profile the actual coupled runner with temporary CUDA-event instrumentation.

An uninstrumented pass reports synchronized physics wall time. A second pass
records GPU stream intervals for nested stages; their overhead is reported
separately. No renderer, MPM solver, training, or physics modification is used.
"""

import argparse
import contextlib
import functools
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import warp as wp

from particle_nerd.data_collection.dataset import initialize_predictor
from particle_nerd.data_collection.trajectories import drag_target_offset
from particle_nerd.force_estimation import body_velocity, kernels
from particle_nerd.runner.simulation import CoupledSimulation
from particle_nerd.transolver import inference
from particle_nerd.transolver.closed_loop import drag_force


class StageEvents:
    def __init__(self, stream):
        self.stream = stream
        self.events = []

    def wrap(self, name, operation):
        @functools.wraps(operation)
        def measured(*args, **kwargs):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(self.stream)
            result = operation(*args, **kwargs)
            end.record(self.stream)
            self.events.append((name, begin, end))
            return result

        return measured

    def collect(self):
        durations = {}
        for name, begin, end in self.events:
            durations[name] = durations.get(name, 0.0) + begin.elapsed_time(end)
        self.events.clear()
        return durations

    @contextlib.contextmanager
    def instrument(self, simulation):
        kernel_names = {
            body_velocity.initialize_body_velocity_solve: "body_velocity_solve",
            kernels.fit_gradients: "velocity_gradient_G",
            kernels.prepare_body: "body_free_motion",
            kernels.prepare_contacts: "contact_geometry_and_response",
            kernels.compact_contacts: "contact_compaction",
            kernels.solve_contacts_parallel: "iterative_impulse_solve",
            inference.update: "particle_state_update",
        }
        launch = wp.launch

        def measured_launch(kernel, *args, **kwargs):
            name = kernel_names.get(kernel)
            return self.wrap(name, launch)(kernel, *args, **kwargs) if name else launch(kernel, *args, **kwargs)

        with contextlib.ExitStack() as stack:
            for owner, attribute, name in [
                (simulation, "step", "complete_physics"),
                (simulation.predictor, "step", "particle_step"),
                (simulation.predictor.model, "forward", "neural_network"),
                (inference, "features", "particle_collider_features"),
                (simulation.estimator, "estimate", "force_estimator"),
                (simulation.estimator.grid, "build", "particle_hash_grid"),
                (wp.utils, "array_scan", "contact_compaction"),
                (wp, "capture_launch", "rigid_four_substeps"),
            ]:
                stack.enter_context(patch.object(owner, attribute, self.wrap(name, getattr(owner, attribute))))
            stack.enter_context(patch.object(wp, "launch", measured_launch))
            yield


def summarize(values):
    return dict(
        mean_ms=float(np.mean(values)), median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95))
    )


def run(args, episode, path, instrumented):
    predictor, initial = initialize_predictor(args.checkpoint, episode)
    # Per-kernel Python instrumentation does not execute on graph replay.
    # Keep the primary wall-time pass captured; use eager NN/estimator launches
    # only for the separately labeled diagnostic breakdown.
    predictor.use_cuda_graph = not instrumented
    simulation = CoupledSimulation(
        predictor,
        initial["manifest"],
        initial["particle_mass"],
        initial["body_pose"],
        initial["body_velocity"],
        effective_mass_scale=180,
        jacobi_relaxation=0.075,
        max_iterations=2000,
        contact_solver=args.contact_solver,
        use_cuda_graph=not instrumented,
    )
    count = args.steps * simulation.substeps
    targets = np.repeat(initial["body_pose"][None, :3], count, axis=0)
    envelopes = np.zeros(count, dtype=np.float32)
    for i, t in enumerate(np.arange(count) * 0.005):
        if 0.25 <= t < 4.75:
            targets[i] += drag_target_offset(path, t - 0.25, 4.5)
            envelopes[i] = min(1, (t - 0.25) / 0.2, (4.75 - t) / 0.2)
    target_gpu = wp.array(targets, dtype=wp.vec3, device=simulation.device)
    envelope_gpu = wp.array(envelopes, dtype=float, device=simulation.device)
    applied = wp.zeros(count, dtype=wp.spatial_vector, device=simulation.device)
    recorder = StageEvents(simulation.torch_stream)
    rows, final_pose = [], None
    with recorder.instrument(simulation) if instrumented else contextlib.nullcontext():
        for step in range(args.steps):

            def controller(state, substep, wrench):
                wp.launch(
                    drag_force,
                    1,
                    [
                        state.body_q,
                        state.body_qd,
                        target_gpu,
                        envelope_gpu,
                        step * simulation.substeps + substep,
                        float(initial["manifest"]["body_mass_kg"]),
                        wrench,
                        applied,
                    ],
                    device=simulation.device,
                )

            wp.synchronize_stream(simulation.stream)
            start = time.perf_counter()
            simulation.step(substep_controller=controller)
            wp.synchronize_stream(simulation.stream)
            wall = (time.perf_counter() - start) * 1000
            gpu = recorder.collect()
            diagnostics = simulation.diagnostics()
            final_pose = diagnostics["body_pose"]
            if not (
                diagnostics["contact"]["converged"]
                and diagnostics["contact"]["finite"]
                and diagnostics["particle"]["finite"]
                and diagnostics["particle"]["velocity_finite"]
                and np.isfinite(final_pose).all()
                and not diagnostics["rigid_contact"]["overflow"]
            ):
                raise RuntimeError(f"Profile encountered invalid/unconverged state: {diagnostics}")
            if step >= args.warmup:
                rows.append(
                    dict(
                        step=step + 1,
                        wall_ms=wall,
                        gpu_ms=gpu,
                        contacts=diagnostics["contact"]["contacts"],
                        iterations=diagnostics["contact"]["iterations"],
                    )
                )
    stages = {key: summarize([row["gpu_ms"][key] for row in rows]) for key in rows[0]["gpu_ms"]}
    return dict(
        neural_cuda_graph=not instrumented,
        estimator_cuda_graph=not instrumented,
        particles=len(predictor.x),
        configuration=predictor.config,
        rigid_contact_capacity=simulation.rigid.solver.mjw_data.naconmax,
        wall=summarize([r["wall_ms"] for r in rows]),
        stages=stages,
        contacts_median=float(np.median([r["contacts"] for r in rows])),
        contacts_range=[min(r["contacts"] for r in rows), max(r["contacts"] for r in rows)],
        iterations_median=float(np.median([r["iterations"] for r in rows])),
        iterations_max=max(r["iterations"] for r in rows),
        final_pose=final_pose,
        measurements=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-solver", choices=("jacobi", "body_velocity"), default="body_velocity")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 1 or args.steps <= args.warmup:
        parser.error("Require steps > warmup >= 1")
    result = dict(
        gpu=torch.cuda.get_device_name(),
        particle_dt_s=0.02,
        rigid_dt_s=0.005,
        contact_solver=args.contact_solver,
        relaxation=0.075,
        iteration_limit=2000,
        tolerance=1e-9,
        effective_mass_scale=180,
        adaptive=False,
        rigid_cuda_graph=True,
        particle_cuda_graph=True,
        particle_capture_scope="Transolver neural forward only; feature generation and state update remain eager",
        estimator_cuda_graph=True,
        steps_per_pass=args.steps,
        discarded_steps=args.warmup,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        cases={},
        timing_scope="Uninstrumented synchronized wall time; separate CUDA-event GPU stream intervals. "
        "First steps excluded; no rendering, diagnostics or disk writes in timings. Nested stages overlap; "
        "event intervals can include launch gaps. Baseline uses neural and estimator graphs; "
        "instrumented pass disables these two graphs to expose individual kernel stages. "
        "Its wall time includes eager launch and instrumentation overhead, not just instrumentation.",
    )
    for path in ("wide-circle", "figure-eight"):
        result["cases"][path] = {}
        for name, instrumented in (("baseline", False), ("instrumented", True)):
            result["cases"][path][name] = run(args, args.reference_dir / f"{path}.hdf5", path, instrumented)
            print(path, name, result["cases"][path][name]["wall"], flush=True)
        base, measured = (result["cases"][path][key] for key in ("baseline", "instrumented"))
        result["particles"] = base["particles"]
        result["cases"][path]["final_pose_max_absolute_difference"] = float(
            np.max(np.abs(np.asarray(base["final_pose"]) - measured["final_pose"]))
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
