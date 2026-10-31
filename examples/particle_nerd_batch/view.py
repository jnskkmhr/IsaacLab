# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive batched simulation with dynamic mouse dragging and brown particles."""

import argparse
import time
from pathlib import Path

import torch
import warp as wp

from .simulation import Simulation


@wp.kernel
def display_points(
    x: wp.array(dtype=wp.vec3), offsets: wp.array(dtype=wp.vec3), particles: int, output: wp.array(dtype=wp.vec3)
):
    i = wp.tid()
    output[i] = x[i] + offsets[i // particles]


def _parse_active_particles(value: str):
    if value.lower() in ("none", "all", "full", "off", "0"):
        return None
    if value.lower() in ("auto", "heuristic"):
        return "auto"
    return int(value)


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
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--effective-mass-scale",
        type=float,
        help="Auxiliary material mass multiplier (default: 180 for shovel, 10 otherwise)",
    )
    parser.add_argument(
        "--active-particles",
        type=_parse_active_particles,
        default="auto",
        help="Active particles per environment ('auto' for proximity+material heuristic, or integer cap)",
    )
    parser.add_argument(
        "--force-model",
        choices=("auto", "analytical", "rft", "neural_contact"),
        default="auto",
        help="Select neural contact head, RFT directional contact solver, or analytical contact estimator",
    )
    parser.add_argument(
        "--env-spacing", type=float, default=4.0, help="Distance between displayed environment origins [m]"
    )
    viz = parser.add_mutually_exclusive_group()
    viz.add_argument("--viz", choices=("newton_gl", "newton_rtx"), default="newton_gl")
    viz.add_argument(
        "--renderer",
        dest="viz",
        type=lambda value: {"gl": "newton_gl", "rtx": "newton_rtx"}.get(value, value),
        choices=("newton_gl", "newton_rtx"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--video", action="store_true", help="Record an MP4 video of the viewer")
    parser.add_argument("--video_length", "--video-length", dest="video_length", type=int, default=None, help="Video length in frames")
    parser.add_argument("--video-output", type=Path, default=Path("/tmp/shovel-batch-video.mp4"), help="Output MP4 path when --video is enabled")
    args = parser.parse_args()
    if args.effective_mass_scale is not None and not 0 < args.effective_mass_scale < float("inf"):
        parser.error("--effective-mass-scale must be finite and positive")
    if not 0 < args.env_spacing < float("inf"):
        parser.error("--env-spacing must be a finite positive distance in meters")
    for path in (args.checkpoint, args.episode):
        if not path.is_file():
            parser.error(f"File not found: {path}. Supply --checkpoint and --episode for your local artifacts.")
    torch.set_num_threads(1)
    sim = Simulation.from_files(
        args.checkpoint,
        args.episode,
        args.num_envs,
        compile_model=args.compile,
        effective_mass_scale=args.effective_mass_scale,
        active_particles=args.active_particles,
        force_model=args.force_model,
    )
    if args.viz == "newton_rtx":
        from newton.viewer import ViewerRTX

        viewer = ViewerRTX(width=1200, height=800, environment="studio", async_rendering=False)
    else:
        from newton.viewer import ViewerGL

        viewer = ViewerGL(width=1200, height=800)
    viewer.set_model(sim.rigid.model)
    viewer.set_world_offsets((args.env_spacing, args.env_spacing, 0.0))
    viewer.set_camera(wp.vec3(1.8, -2.4, 1.8))
    viewer.camera.look_at((0, 0, 0.18))
    positions = wp.empty_like(sim.x)
    colors = wp.full(len(sim.x), wp.vec3(0.32, 0.24, 0.21), dtype=wp.vec3, device=sim.device)
    radii = wp.full(len(sim.x), 0.012, dtype=wp.float32, device=sim.device)
    mouse = sim.rigid.model.state()
    points_initialized = False
    video_writer = None
    if args.video:
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
                str(round(1.0 / sim.dt)),
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
    recorded_frames = 0

    def draw():
        nonlocal points_initialized, recorded_frames, video_writer
        with torch.cuda.stream(sim.torch_stream), wp.ScopedStream(sim.stream):
            wp.launch(
                display_points, len(sim.x), [sim.x, viewer.world_offsets, sim.particles, positions], device=sim.device
            )
            viewer.begin_frame(sim.steps * sim.dt)
            viewer.log_state(sim.rigid.states[sim.parity])
            if not points_initialized:
                viewer.log_points("sand", positions, radii=0.012, colors=colors)
                points_initialized = True
            elif args.viz == "newton_rtx":
                viewer.log_points("sand", positions)
            else:
                viewer.log_points("sand", positions, radii=radii)
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

    draw()
    print("Right-click and drag an intruder to apply force. Space pauses/resumes. Worlds are independent.", flush=True)
    step_limit = args.steps if args.steps is not None else args.video_length
    try:
        while viewer.is_running() and (step_limit is None or sim.steps < step_limit):
            start = time.perf_counter()
            if viewer.should_step():
                with torch.cuda.stream(sim.torch_stream), wp.ScopedStream(sim.stream):
                    current = sim.rigid.states[sim.parity]
                    wp.copy(mouse.body_q, current.body_q)
                    wp.copy(mouse.body_qd, current.body_qd)
                    mouse.clear_forces()
                    viewer.apply_forces(mouse)
                sim.step(wp.to_torch(mouse.body_f))
            draw()
            time.sleep(max(0, sim.dt - (time.perf_counter() - start)))
    finally:
        if video_writer is not None:
            video_writer.stdin.close()
            video_writer.wait()
            print(f"Saved {recorded_frames} frames to {args.video_output}", flush=True)
        viewer.close()


if __name__ == "__main__":
    main()
