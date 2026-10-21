# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render recorded G1 runs in Newton RTX, with synchronized foot-force profiles."""

import argparse
import io
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

from particle_nerd.g1.collect import configure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--prediction", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    paths = [args.reference] + ([args.prediction] if args.prediction else [])
    data = []
    for path in paths:
        meta = json.loads((path / "metadata.json").read_text())
        data.append((meta, dict(np.load(path / "snapshots.npz")), dict(np.load(path / "trace.npz"))))
    import gymnasium as gym
    from isaaclab_newton.physics import NewtonManager
    from newton.viewer import ViewerRTX

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    task = "IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play"
    cfg = configure(parse_env_cfg(task, num_envs=1), 1)
    cfg.sim.physics.load_visual_shapes = True
    width, height, top, chart_height = 640 * len(data), 480, 80, 240
    fig, axes = plt.subplots(1, 2, figsize=(width / 100, chart_height / 100), dpi=100)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.86, wspace=0.40)
    colors = ["#365e7c", "#bc703d"]
    labels = ["MPM reference", "Particle NeRD + analytical forces"]
    for i, (_, _, trace) in enumerate(data):
        for foot, ax in enumerate(axes):
            ax.plot(trace["time_s"], trace["foot_wrench"][:, foot, 2], label=labels[i], color=colors[i], lw=1.2)
            ax.set(xlabel="Time [s]", ylabel="Sand force on foot [N]", title=["Left foot", "Right foot"][foot])
            ax.tick_params(labelsize=8)
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_size(9)
            ax.title.set_size(10)
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.canvas.draw()
    bounds = [ax.get_window_extent().bounds for ax in axes]
    limits = [ax.get_xlim() for ax in axes]
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    buffer.seek(0)
    chart = Image.open(buffer).convert("RGB")
    total_height = top + height + chart_height
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg.sim, dict(headless=True, device="cuda:0", visualizer=[])):
        env = gym.make(task, cfg=cfg)
        viewer, writer = None, None
        try:
            model = NewtonManager.get_model()
            for meta, _, _ in data:
                if list(model.body_label) != meta["body_labels"] or model.particle_count != meta["particle_count"]:
                    raise ValueError("Replay geometry does not match recorded bodies/particles")
            state = model.state()
            viewer = ViewerRTX(
                width=640, height=height, headless=True, fps=25, environment="studio", async_rendering=False
            )
            viewer.set_model(model)
            viewer.show_particles = False
            viewer.show_collision = False
            viewer.show_visual = True
            viewer.set_camera(wp.vec3(3.5, -4, 2.4))
            viewer.camera.look_at((0, 0, 0.15))
            particle_colors = wp.full(
                model.particle_count, wp.vec3(0.32, 0.24, 0.21), dtype=wp.vec3, device=model.device
            )
            frames = min(len(d[1]["time_s"]) for d in data)
            indices = [min(50, frames - 1)] if args.preview else range(frames)
            if not args.preview:
                writer = subprocess.Popen(
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
                        f"{width}x{total_height}",
                        "-r",
                        "25",
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "21",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(args.output),
                    ],
                    stdin=subprocess.PIPE,
                )
            for k in indices:
                frame = Image.new("RGB", (width, total_height), "#f7f9f7")
                draw = ImageDraw.Draw(frame)
                for i, (meta, snap, _) in enumerate(data):
                    t = float(snap["time_s"][k])
                    state.body_q.assign(snap["body_pose"][k])
                    state.particle_q.assign(snap["particle_position"][k])
                    for _ in range(8 if k == indices[0] else 2):
                        viewer.begin_frame(t)
                        viewer.log_state(state)
                        viewer.log_points("sand", state.particle_q, radii=0.010, colors=particle_colors)
                        viewer.end_frame()
                    frame.paste(Image.fromarray(viewer.get_frame().numpy()).convert("RGB"), (i * 640, top))
                    draw.text((i * 640 + 14, 12), labels[i], font=font, fill=colors[i])
                    draw.text(
                        (i * 640 + 14, 35), f"Dynamic G1 | 3 x 3 m sand | t = {t:.2f} s", font=font, fill="#23383e"
                    )
                    outcome = meta["outcome"].replace("_", " ")
                    status = f"Recorded run: {meta['duration_s']:.2f} s ({outcome})"
                    draw.text((i * 640 + 14, 59), status, font=small_font, fill="#566970")
                frame.paste(chart, (0, top + height))
                for (x, y, w, h), (xmin, xmax) in zip(bounds, limits, strict=True):
                    px = x + (t - xmin) / max(xmax - xmin, 0.01) * w
                    draw.line(
                        (px, top + height + chart_height - y - h, px, top + height + chart_height - y),
                        fill="#24796d",
                        width=2,
                    )
                if k == indices[0] or k == min(50, frames - 1):
                    frame.save(args.output.with_suffix(".jpg"), quality=90)
                if writer:
                    writer.stdin.write(frame.tobytes())
                if k % 25 == 0:
                    print("RENDER", k, frames, flush=True)
            if writer:
                writer.stdin.close()
                if writer.wait() != 0:
                    raise RuntimeError("ffmpeg failed")
                writer = None
        finally:
            if writer:
                writer.stdin.close()
                writer.wait()
            if viewer:
                viewer.close()
            env.close()
    print("RENDER_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
