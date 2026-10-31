# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run learned particles, analytical contact forces and a mouse-draggable rigid body; no MPM."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from particle_nerd.data_collection.dataset import initialize_predictor
from particle_nerd.runner.simulation import CoupledSimulation
from particle_nerd.runner.viewer import create_viewer, draw_scene, particle_colors


def main(*, default_checkpoint=None, default_episode=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint, required=default_checkpoint is None)
    parser.add_argument(
        "--episode",
        type=Path,
        default=default_episode,
        required=default_episode is None,
        help="Only the settled initial state/history is read",
    )
    parser.add_argument("--selector", type=Path)
    parser.add_argument(
        "--bed-size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Transolver only: fresh particle block in metres, at training spacing; starts at rest above rigid ground",
    )
    parser.add_argument("--log-every", type=int, default=0, help="Print diagnostics every N steps (default: disabled)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--show-contact-forces",
        action="store_true",
        help="Initially show material force arrows; also toggle in viewer Rendering Options",
    )
    parser.add_argument("--viz", choices=("newton_gl", "newton_rtx"), default="newton_gl")
    parser.add_argument("--steps", type=int, help="Step limit; default: until window closes, or 250 headless")
    parser.add_argument(
        "--max-iterations", type=int, help="Iteration safety limit (default: 2000); stops earlier on convergence"
    )
    parser.add_argument("--effective-mass-scale", type=float, help="Default: 180 for shovel, 10 for box")
    parser.add_argument("--contact-relaxation", type=float, help="Default: 0.075 for shovel, 0.1 for box")
    parser.add_argument(
        "--contact-solver",
        choices=("jacobi", "body_velocity", "rft_body_velocity"),
        help="Default: body_velocity for shovel, jacobi otherwise",
    )
    parser.add_argument("--contact-warm-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--video", action="store_true", help="Record an MP4 video of the viewer")
    parser.add_argument("--video_length", "--video-length", dest="video_length", type=int, default=None, help="Video length in frames")
    parser.add_argument("--video-output", type=Path, default=Path("/tmp/shovel-video.mp4"), help="Output MP4 path when --video is enabled")
    parser.add_argument("--output", type=Path, help="Optional diagnostic JSON log")
    args = parser.parse_args()
    if args.log_every < 0:
        parser.error("--log-every must be nonnegative")
    if args.bed_size is not None and (not np.isfinite(args.bed_size).all() or min(args.bed_size) <= 0):
        parser.error("--bed-size dimensions must be finite and positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    for path in (args.checkpoint, args.episode):
        if not path.is_file():
            parser.error(f"File not found: {path}. Supply --checkpoint and --episode for your local artifacts.")
    wp.config.log_level = wp.LOG_WARNING
    predictor, initial = initialize_predictor(
        args.checkpoint, args.episode, args.selector, args.compile, bed_size=args.bed_size
    )
    shovel = initial["manifest"].get("geometry") == "shovel"
    max_iterations = args.max_iterations if args.max_iterations is not None else 2000
    mass_scale = args.effective_mass_scale if args.effective_mass_scale is not None else (180.0 if shovel else 10.0)
    relaxation = args.contact_relaxation if args.contact_relaxation is not None else (0.075 if shovel else 0.1)
    step_limit = (
        args.steps
        if args.steps is not None
        else (args.video_length if args.video_length is not None else (250 if args.headless else None))
    )
    contact_solver = args.contact_solver or ("body_velocity" if shovel and not args.contact_warm_start else "jacobi")
    if args.contact_warm_start and contact_solver == "body_velocity":
        parser.error("--contact-warm-start requires --contact-solver jacobi")
    simulation = CoupledSimulation(
        predictor,
        initial["manifest"],
        initial["particle_mass"],
        initial["body_pose"],
        initial["body_velocity"],
        max_iterations=max_iterations,
        effective_mass_scale=mass_scale,
        jacobi_relaxation=relaxation,
        contact_warm_start=args.contact_warm_start,
        contact_solver=contact_solver,
    )
    viewer = None
    if not args.headless:
        viewer = create_viewer(args.viz, simulation.rigid.model)
        colors = particle_colors(len(predictor.x), predictor.device)
        mouse_forces = simulation.rigid.model.state()
        from particle_nerd.runner.force_visualizer import ContactForceVisualizer

        with wp.ScopedStream(simulation.stream):
            force_visualizer = ContactForceVisualizer(viewer, simulation, args.show_contact_forces)
            viewer.begin_frame(0.0)
            draw_scene(viewer, simulation.rigid.state, wp.from_torch(predictor.x, dtype=wp.vec3), colors)
            viewer.end_frame()
    print(
        "Coupled runner: learned particles + analytical contact + MJWarp; no MPM or future body motion",
        flush=True,
    )
    print(f"Contact settings: effective mass x{mass_scale:g}, relaxation {relaxation:g}, limit {max_iterations}")
    print(f"Contact solver: {contact_solver}")
    print(f"Contact warm start: {args.contact_warm_start}")
    if args.bed_size is not None:
        print(
            f"Fresh particle field: {args.bed_size} m, {len(predictor.x):,} particles; checkpoint accuracy unvalidated"
        )
    if viewer is not None:
        print(
            "Right-click and drag the shovel/body to apply force. Space: pause/resume. Close window: quit.", flush=True
        )
    rows = []
    step = 0
    contact_count = 0
    video_writer = None
    recorded_frames = 0
    if args.video and viewer is not None:
        import subprocess

        args.video_output.parent.mkdir(parents=True, exist_ok=True)
        video_writer = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1200x800",
                "-r",
                str(round(1.0 / predictor.dt)),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.video_output),
            ],
            stdin=subprocess.PIPE,
        )
        print(f"Recording video to {args.video_output} ...", flush=True)
    try:
        while step_limit is None or step < step_limit:
            frame_started = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                break
            if viewer is None or viewer.should_step():
                if viewer is not None:
                    with wp.ScopedStream(simulation.stream):
                        wp.copy(force_visualizer.pose, simulation.rigid.state.body_q)
                        wp.copy(mouse_forces.body_q, simulation.rigid.state.body_q)
                        wp.copy(mouse_forces.body_qd, simulation.rigid.state.body_qd)
                        mouse_forces.clear_forces()
                        viewer.apply_forces(mouse_forces)
                    simulation.step(mouse_forces.body_f)
                else:
                    simulation.step()
                step += 1
                report = simulation.diagnostics()
                contact_count = report["contact"]["contacts"]
                if args.output:
                    rows.append(report)
                failures = []
                if report["particle"].get("graph_overflow", False):
                    failures.append("particle graph capacity exceeded")
                if not report["particle"]["finite"]:
                    failures.append("nonfinite particle position")
                if not report["particle"]["velocity_finite"]:
                    failures.append("nonfinite particle velocity")
                if not report["contact"]["finite"]:
                    failures.append("nonfinite estimated contact wrench")
                if not np.isfinite(report["body_pose"]).all():
                    failures.append("nonfinite rigid-body pose")
                if not np.isfinite(report["body_velocity"]).all():
                    failures.append("nonfinite rigid-body velocity")
                if report["rigid_contact"]["generated"] > report["rigid_contact"]["capacity"]:
                    failures.append("rigid-contact capacity exceeded")
                if report["rigid_contact"]["overflow"]:
                    failures.append("MJWarp solver buffer overflow")
                if failures or (args.log_every and step % args.log_every == 0):
                    print(json.dumps(report), flush=True)
                if failures:
                    raise RuntimeError(f"Invalid coupled state at step {step}: {'; '.join(failures)}")
            if viewer is not None:
                with wp.ScopedStream(simulation.stream):
                    viewer.begin_frame(step * predictor.dt)
                    draw_scene(viewer, simulation.rigid.state, wp.from_torch(predictor.x, dtype=wp.vec3), colors)
                    force_visualizer.draw(viewer, contact_count)
                    viewer.end_frame()
                if video_writer is not None:
                    frame_np = viewer.get_frame().numpy()[..., :3]
                    video_writer.stdin.write(frame_np.tobytes())
                    recorded_frames += 1
                    if args.video_length is not None and recorded_frames >= args.video_length:
                        video_writer.stdin.close()
                        video_writer.wait()
                        video_writer = None
                        print(f"Saved {recorded_frames} frames to {args.video_output}", flush=True)
                remaining = predictor.dt - (time.perf_counter() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if video_writer is not None:
            video_writer.stdin.close()
            video_writer.wait()
            print(f"Saved {recorded_frames} frames to {args.video_output}", flush=True)
        if viewer is not None:
            viewer.close()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    dict(mode="experimental-coupled", initial_scene=initial["manifest"]["scene"], steps=rows), indent=2
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
