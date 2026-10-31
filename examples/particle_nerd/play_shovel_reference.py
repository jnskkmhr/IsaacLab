# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM–MJWarp reference with the same scene and viewer as Particle NeRD play."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp


def main():
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.sim.spawners.mpm import MPMPointsCfg

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.envs import ManagerBasedEnv

    from particle_nerd.data_collection.collect import (
        configure_body_mass,
        create_collection_env_cfg,
        validate_rigid_asset,
    )
    from particle_nerd.data_collection.dataset import initialize_fresh_bed, load_initial_state
    from particle_nerd.runner.rigid import build_rigid_model, rigid_scene_description
    from particle_nerd.runner.viewer import create_viewer, draw_scene, particle_colors

    artifacts = Path(__file__).resolve().parents[3] / "particle_nerd"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode", type=Path, default=artifacts / "experiments/particle_gnn/shovel/reference/wide-circle.hdf5"
    )
    parser.add_argument("--bed-size", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel MPM environments")
    parser.add_argument(
        "--env-spacing", type=float, default=4.0, help="Distance between displayed environment origins [m]"
    )
    parser.add_argument("--steps", type=int, help="20 ms frame limit; otherwise run until the viewer closes")
    parser.add_argument("--video", action="store_true", help="Record an MP4 video of the viewer")
    parser.add_argument("--video_length", "--video-length", dest="video_length", type=int, default=None, help="Video length in frames")
    parser.add_argument("--video-output", type=Path, default=Path("/tmp/shovel-mpm-video.mp4"), help="Output MP4 path when --video is enabled")
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--headless", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if not 0 < args.env_spacing < float("inf"):
        parser.error("--env-spacing must be a finite positive distance in meters")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if not args.episode.is_file():
        parser.error(f"Episode not found: {args.episode}")
    if args.bed_size is not None and (not np.isfinite(args.bed_size).all() or min(args.bed_size) <= 0):
        parser.error("--bed-size dimensions must be finite and positive")
    if len(args.visualizer) != 1 or args.visualizer[0] not in ("newton_gl", "newton_rtx"):
        parser.error("Use --viz newton_gl or --viz newton_rtx; --headless disables the viewer")
    wp.config.log_level = wp.LOG_WARNING
    initial = load_initial_state(args.episode)
    if initial["manifest"].get("geometry") != "shovel":
        parser.error("The reference launcher requires a shovel episode")
    if args.bed_size is not None:
        initialize_fresh_bed(initial, args.bed_size)
    args.geometry, args.body_mass = "shovel", None
    asset_metadata = configure_body_mass(args, parser)
    manifest = initial["manifest"]
    cfg, _ = create_collection_env_cfg("shovel", args.body_mass, args.seed, bed_size=manifest["scene"]["bed_size_m"])
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = args.env_spacing
    material = cfg.scene.sand.spawn.material
    for name, value in manifest["material"].items():
        setattr(material, name, value)
    # Explicit points preserve the exact visible initial state, ordering, mass,
    # and sampling used by NeRD. The episode does not contain MPM stress/history;
    # those solver fields start reset, rather than being reconstructed from x/v.
    cfg.scene.sand.spawn = MPMPointsCfg(
        positions=initial["position"].tolist(),
        velocities=initial["velocity"].tolist(),
        mass=initial["particle_mass"].tolist(),
        radius=initial["particle_radius"].tolist(),
        material=material,
    )
    cfg.events.reset_material.params["jitter_fraction"] = 0.0
    cfg.events.randomize_box = None
    cfg.sim.physics.solver_cfg.entries[0].solver_cfg.njmax = None
    cfg.sim.physics.use_cuda_graph = True
    # PIC integration and zero grid padding need at most one active cell per
    # particle. Persistent capacity lets Newton rebuild the sparse grid on GPU
    # inside the captured coupled step; the manager checks overflow on replay.
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.max_active_cell_count = len(initial["position"]) * args.num_envs
    # Physics and manager terms still come from the collection pipeline. Both
    # play scripts share the same native viewer, rather than separate USD scenes.
    runtime_args = dict(vars(args), visualizer=[])
    with launch_simulation(cfg, runtime_args):
        env = ManagerBasedEnv(cfg)
        viewer = None
        try:
            model = NewtonManager.get_model()
            validate_rigid_asset(model, args, asset_metadata)
            np.testing.assert_allclose(
                model.particle_mass.numpy(), np.tile(initial["particle_mass"], args.num_envs), rtol=1e-6
            )
            env.reset(seed=args.seed)
            pose = torch.as_tensor(initial["body_pose"], dtype=torch.float32, device=env.device).repeat(
                args.num_envs, 1
            )
            pose[:, :3] += env.scene.env_origins
            vel = torch.as_tensor(initial["body_velocity"], dtype=torch.float32, device=env.device).repeat(
                args.num_envs, 1
            )
            env.scene["body5"].write_root_pose_to_sim_index(root_pose=pose)
            env.scene["body5"].write_root_velocity_to_sim_index(root_velocity=vel)
            NewtonManager.forward()
            command = torch.zeros((args.num_envs, 6), device=env.device)
            frame_dt = 1.0 / manifest["record_hz"]
            substeps = round(frame_dt / cfg.sim.dt)
            if not np.isclose(substeps * cfg.sim.dt, frame_dt):
                raise ValueError("Episode interval must be a whole number of MPM steps")
            colors = particle_colors(model.particle_count, model.device)
            if not args.headless:
                display_model = build_rigid_model(
                    rigid_scene_description(manifest, pose.cpu().numpy(), vel.cpu().numpy()), model.device
                )
                viewer = create_viewer(args.visualizer[0], display_model)
                # Display and physics share identical dynamic shovel body indices.
                # Apply native picking to the actual coupled physics state.
                NewtonManager.register_state_force_callback(viewer.apply_forces)
                viewer.begin_frame(0.0)
                state = NewtonManager.get_state_0()
                draw_scene(viewer, state, state.particle_q, colors)
                viewer.end_frame()
            print(
                f"MPM + MJWarp reference: {args.num_envs} worlds, {model.particle_count:,} particles; "
                "shared NeRD scene/viewer",
                flush=True,
            )
            print("Right-click drag: apply force. Space: pause/resume. Close window: quit.", flush=True)
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
                        str(round(1.0 / frame_dt)),
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
            steps = 0
            limit = (
                args.steps
                if args.steps is not None
                else (args.video_length if args.video_length is not None else (250 if args.headless else None))
            )
            with torch.inference_mode():
                while limit is None or steps < limit:
                    started = time.perf_counter()
                    if viewer is not None and not viewer.is_running():
                        break
                    if viewer is None or viewer.should_step():
                        for _ in range(substeps):
                            env.step(command)
                        steps += 1
                    if viewer is not None:
                        viewer.begin_frame(steps * frame_dt)
                        state = NewtonManager.get_state_0()
                        draw_scene(viewer, state, state.particle_q, colors)
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
                        remaining = frame_dt - (time.perf_counter() - started)
                        if remaining > 0:
                            time.sleep(remaining)
        finally:
            if video_writer is not None:
                video_writer.stdin.close()
                video_writer.wait()
                print(f"Saved {recorded_frames} frames to {args.video_output}", flush=True)
            if viewer is not None:
                viewer.close()
            env.close()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
