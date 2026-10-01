# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record prescribed NPZ motion in Newton, with no policy or physics steps.

Keep this file beside convert_lafan_newton.py. Checks FK consistency against
the saved NPZ; this is not an independent validation of the source retargeting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from convert_lafan_newton import SUPPORTED_COMMITS, TASK, validate_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    repo = opts.repo.expanduser().resolve()
    motion_path = opts.motion.expanduser().resolve()
    video = opts.output.expanduser().resolve()
    if video.suffix.lower() != ".mp4":
        parser.error("--output must end in .mp4")
    if video.exists():
        raise FileExistsError(f"Choose a new output filename: {video}")
    if video.with_suffix(".partial.mp4").exists():
        raise FileExistsError("A partial video exists; choose a new --output filename")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head not in SUPPORTED_COMMITS:
        raise RuntimeError(f"Unreviewed commit: {head}; supported: {sorted(SUPPORTED_COMMITS)}")
    if Path(sys.executable).absolute().parent != repo / ".venv/bin":
        raise RuntimeError("Run with uv run --frozen --no-sync python in this checkout")
    dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--", "source", "scripts"], text=True)
    if dirty.strip():
        raise RuntimeError("Review tracked code changes before using this version-specific player")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError("No ffmpeg executable or installed imageio_ffmpeg found") from exc
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    print("REFERENCE_STAGE: importing Newton dependencies", flush=True)
    import warp as wp

    wp.config.enable_backward = False
    import numpy as np
    import torch
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import build_simulation_context

    import isaaclab_tasks
    from isaaclab_tasks.utils import resolve_task_config

    if not Path(isaaclab_tasks.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Wrong task installation imported")
    digest = hashlib.sha256(motion_path.read_bytes()).hexdigest()
    with np.load(motion_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    frames = validate_arrays(arrays)
    fps = 60
    source_joints = arrays["joint_names"].tolist()
    source_bodies = arrays["body_names"].tolist()
    source_pelvis = source_bodies.index("pelvis")
    cfg, _ = resolve_task_config(TASK, None, overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = 1
    cfg.scene.contact_forces = None
    cfg.sim.device = "cuda:0"
    cfg.sim.render_interval = cfg.decimation
    cfg.video_recorders = []
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            headless=True,
            streaming_view=False,
            window_width=960,
            window_height=540,
            enable_picking=False,
            show_collision=False,
            eye=(3.0, -3.0, 2.0),
            lookat=(0.0, 0.0, 0.8),
        )
    ]
    launcher_parser = argparse.ArgumentParser()
    add_launcher_args(launcher_parser)
    launcher = launcher_parser.parse_args(["--device", "cuda:0", "--viz", "newton_gl"])
    video.parent.mkdir(parents=True, exist_ok=True)
    partial = video.with_suffix(".partial.mp4")
    report = dict(
        commit=head,
        motion=str(motion_path),
        motion_sha256=digest,
        kind="Prescribed kinematic reference; no policy, no physics integration",
        reference_fps=fps,
        video_fps=30,
        reference_frames=frames,
        max_body_position_error_m=0.0,
        max_body_rotation_error_rad=0.0,
        max_joint_write_error_rad=0.0,
        video_frames=0,
        independent_retargeting_validation=False,
        visual_review_completed=False,
    )

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else value.torch

    print("REFERENCE_STAGE: creating robot and headless video renderer", flush=True)
    with launch_simulation(cfg, launcher):
        if "newton" not in type(cfg.sim.physics).__module__.lower():
            raise RuntimeError("Expected Newton backend")
        with build_simulation_context(sim_cfg=cfg.sim) as sim:
            scene = InteractiveScene(cfg.scene)
            sim.reset()
            scene.reset()
            robot = scene["robot"]
            names, bodies = list(robot.joint_names), list(robot.body_names)
            if len(names) != len(source_joints) or set(names) != set(source_joints):
                raise RuntimeError("Robot and NPZ joints differ")
            if len(bodies) != len(source_bodies) or set(bodies) != set(source_bodies):
                raise RuntimeError("Robot and NPZ bodies differ")
            if bodies[0] != "pelvis":
                raise RuntimeError("Expected pelvis root")
            joint_order = [source_joints.index(n) for n in names]
            body_order = [source_bodies.index(n) for n in bodies]
            joint_pos = torch.as_tensor(arrays["joint_pos"][:, joint_order], device=sim.device)
            joint_vel = torch.as_tensor(arrays["joint_vel"][:, joint_order], device=sim.device)
            root = np.concatenate(
                [arrays["body_pos_w"][:, source_pelvis], arrays["body_quat_w"][:, source_pelvis]], axis=-1
            )
            root_gpu = torch.as_tensor(root.copy(), device=sim.device)
            root_gpu[:, :3] += scene.env_origins[0]
            viewers = [v for v in sim.visualizers if v.cfg.visualizer_type == "newton_gl"]
            if len(viewers) != 1:
                raise RuntimeError("Newton GL renderer did not initialize")
            steps_before = sim.get_physics_step_count()
            with video.with_suffix(".ffmpeg.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        ffmpeg,
                        "-n",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        "960x540",
                        "-r",
                        "30",
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(partial),
                    ],
                    stdin=subprocess.PIPE,
                    stderr=log,
                )
                try:
                    with torch.inference_mode():
                        for i in range(frames):
                            robot.write_root_link_pose_to_sim_index(root_pose=root_gpu[i : i + 1])
                            robot.write_joint_state_to_sim_index(
                                position=joint_pos[i : i + 1], velocity=joint_vel[i : i + 1]
                            )
                            sim.forward()
                            robot.update(sim.get_physics_dt())
                            p = (tensor(robot.data.body_pos_w)[0] - scene.env_origins[0]).cpu().numpy()
                            q = tensor(robot.data.body_quat_w)[0].cpu().numpy()
                            ref_p = arrays["body_pos_w"][i, body_order]
                            ref_q = arrays["body_quat_w"][i, body_order]
                            position_error = float(np.linalg.norm(p - ref_p, axis=-1).max())
                            dots = np.abs(np.sum(q * ref_q, axis=-1))
                            dots /= np.linalg.norm(q, axis=-1) * np.linalg.norm(ref_q, axis=-1)
                            rotation_error = float((2 * np.arccos(dots.clip(0, 1))).max())
                            joint_error = float((tensor(robot.data.joint_pos)[0] - joint_pos[i]).abs().max())
                            if not np.isfinite([position_error, rotation_error, joint_error]).all():
                                raise RuntimeError(f"Nonfinite state at frame {i}")
                            if position_error > 1e-3 or rotation_error > 0.005 or joint_error > 1e-4:
                                raise RuntimeError(
                                    f"NPZ/FK mismatch at {i}: {position_error}, {rotation_error}, {joint_error}"
                                )
                            report["max_body_position_error_m"] = max(
                                report["max_body_position_error_m"], position_error
                            )
                            report["max_body_rotation_error_rad"] = max(
                                report["max_body_rotation_error_rad"], rotation_error
                            )
                            report["max_joint_write_error_rad"] = max(report["max_joint_write_error_rad"], joint_error)
                            if i % 2 == 0:
                                center = root_gpu[i, :3].cpu().numpy().copy()
                                center[2] = 0.8
                                sim.set_camera_view(tuple(center + [3.0, -3.0, 1.4]), tuple(center))
                                sim.render()
                                rgb = viewers[0].render_rgb_array()
                                if rgb is None or rgb.shape != (540, 960, 3) or rgb.dtype != np.uint8:
                                    raise RuntimeError("Invalid RGB frame from Newton GL")
                                process.stdin.write(np.ascontiguousarray(rgb).tobytes())
                                report["video_frames"] += 1
                            if i % 120 == 0:
                                print(f"REFERENCE_FRAME {i}/{frames}", flush=True)
                finally:
                    try:
                        process.stdin.close()
                    finally:
                        try:
                            code = process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                            raise
                if code:
                    raise RuntimeError(f"ffmpeg failed ({code}); see {video.with_suffix('.ffmpeg.log')}")
            report["physics_steps"] = sim.get_physics_step_count() - steps_before
            if report["physics_steps"] != 0:
                raise RuntimeError("Unexpected physics steps during prescribed playback")
    if hashlib.sha256(motion_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Input motion changed during playback")
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError("No video produced")
    partial.replace(video)
    report["video"] = str(video)
    video.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("KINEMATIC_REFERENCE_VIDEO_OK:", video, flush=True)
    print("FK_CONSISTENCY:", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
