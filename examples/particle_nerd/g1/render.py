# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render coupled G1 runs or particle predictions with recorded G1 motion with Newton OpenGL or RTX."""

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


def set_reference_force_limits(axes, reference_trace):
    """Use one force range from the reference MPM run for both feet and both models."""
    force = reference_trace["foot_wrench"][:, :, 2]
    low, high = min(0.0, float(force.min())), max(0.0, float(force.max()))
    padding = 0.05 * max(high - low, 1.0)
    for ax in axes:
        ax.set_ylim(low - padding, high + padding)
    print("MPM_GRF_Y_LIMITS", low - padding, high + padding, flush=True)


def native_view_settings(cfg, viz):
    """Read task-owned resolution; only select the requested headless renderer."""
    native_rtx = viz == "newton_rtx"
    if native_rtx:
        cfg.sim.visualizer_cfgs = [v for v in cfg.sim.visualizer_cfgs if v.visualizer_type == "newton_rtx"]
        visual = cfg.sim.visualizer_cfgs[0]
        visual.headless = True

    selected_visual = next(v for v in cfg.sim.visualizer_cfgs if v.visualizer_type == viz)
    selected_visual.origin_type = "world"
    selected_visual.origin_track_path = None
    panel_width, height = selected_visual.window.size
    return native_rtx, panel_width, height


def main():
    import sys

    frozen_tasks = (
        Path.home()
        / "isaac/particle_nerd/experiments/recurrent-05s-20261021/g1-source-stage/source/source/isaaclab_tasks"
    )
    if frozen_tasks.is_dir() and str(frozen_tasks) not in sys.path:
        sys.path.insert(0, str(frozen_tasks))

    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reference", type=Path)
    inputs.add_argument("--particle-evaluation", type=Path, help="Particle-only evaluation directory")
    p.add_argument("--episode", type=Path, help="Override the evaluation source HDF5 path")
    p.add_argument("--prediction", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--viz", choices=("newton_gl", "newton_rtx"), default="newton_rtx")
    args = p.parse_args()
    data = []
    geometry = None
    if args.particle_evaluation:
        if args.prediction:
            p.error("--prediction cannot be combined with --particle-evaluation")
        import h5py

        report = json.loads((args.particle_evaluation / "metrics.json").read_text())
        snap = dict(np.load(args.particle_evaluation / "snapshots.npz"))
        with h5py.File(args.episode or report["episode"], "r") as episode:
            recorded_times = episode["time_s"][:]
            for k, time in enumerate(snap["time_s"]):
                row = int(np.argmin(abs(recorded_times - time)))
                np.testing.assert_allclose(recorded_times[row], time, atol=1e-6, rtol=0)
                np.testing.assert_allclose(episode["body_pose"][row], snap["body_pose"][k], atol=1e-6, rtol=0)
                np.testing.assert_allclose(
                    episode["particle_position"][row], snap["reference_position"][k], atol=1e-6, rtol=0
                )
            geometry = {
                key: episode[key][:]
                for key in (
                    "body_com_local",
                    "collider_body",
                    "collider_local_pose",
                    "collider_half_size",
                    "foot_body_ids",
                )
            }
        meta = {"particle_count": snap["particle_position"].shape[1]}
        reference = {**snap, "particle_position": snap["reference_position"]}
        data = [(meta, reference, None), (meta, snap, None)]
    else:
        if args.episode:
            p.error("--episode requires --particle-evaluation")
        paths = [args.reference] + ([args.prediction] if args.prediction else [])
        for path in paths:
            meta = json.loads((path / "metadata.json").read_text())
            data.append((meta, dict(np.load(path / "snapshots.npz")), dict(np.load(path / "trace.npz"))))
    import gymnasium as gym
    from isaaclab_newton.physics import NewtonManager
    from newton.viewer import ViewerGL

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    task = "IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play"
    cfg = configure(parse_env_cfg(task, num_envs=1), 1)
    cfg.sim.physics.load_visual_shapes = True
    top, chart_height = 80, 240
    native_rtx, panel_width, height = native_view_settings(cfg, args.viz)
    width = panel_width * len(data)
    fig, axes = plt.subplots(1, 2, figsize=(width / 100, chart_height / 100), dpi=100)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.86, wspace=0.40)
    colors = ["#365e7c", "#bc703d"]
    labels = [
        "MPM reference",
        {
            "transolver_total_body_impulse": "Particle NeRD + learned body impulses",
            "transolver_physics_contact_impulse": "Particle NeRD + physics-structured contact head",
        }.get(
            (data[1][0].get("estimator") or {}).get("method") if len(data) > 1 else None,
            "Particle NeRD + analytical forces",
        ),
    ]
    if args.particle_evaluation:
        labels = ["MPM particles | recorded G1", "Particle NeRD | same recorded G1"]
        rows = report["metrics"]
        times = [row["time_s"] for row in rows]
        for key, label, color in (
            ("position_rmse_m", "Particle NeRD", colors[1]),
            ("frozen_rmse_m", "Fixed initial particle positions", colors[0]),
        ):
            axes[0].plot(times, [100 * row[key] for row in rows], label=label, color=color, lw=1.2)
        axes[0].set(title="All particles", ylabel="Position RMSE [cm]")
        axes[1].plot(
            times,
            [100 * row["stationary_rmse_m"] if row["stationary_rmse_m"] is not None else np.nan for row in rows],
            color=colors[1],
            lw=1.2,
        )
        axes[1].set(title="Particles stationary in MPM reference", ylabel="Position RMSE [cm]")
    else:
        for i, (_, _, trace) in enumerate(data):
            for foot, ax in enumerate(axes):
                ax.plot(trace["time_s"], trace["foot_wrench"][:, foot, 2], label=labels[i], color=colors[i], lw=1.2)
                ax.set(ylabel="Sand force on foot [N]", title=["Left foot", "Right foot"][foot])
        set_reference_force_limits(axes, data[0][2])
    for ax in axes:
        ax.set_xlabel("Recorded time [s]" if args.particle_evaluation else "Time [s]")
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
    with launch_simulation(
        cfg.sim, dict(headless=True, device="cuda:0", visualizer=["newton_rtx"] if native_rtx else [])
    ):
        env = gym.make(task, cfg=cfg)
        viewer, writer = None, None
        try:
            model = NewtonManager.get_model()
            if geometry is not None:
                import newton

                np.testing.assert_allclose(model.body_com.numpy(), geometry["body_com_local"], atol=1e-6)
                feet = env.unwrapped._foot_body_ids.cpu().numpy()[0]
                np.testing.assert_array_equal(feet, geometry["foot_body_ids"])
                eligible = set(feet) | {i for i, label in enumerate(model.body_label) if "/MPM" in label}
                ids = [
                    i
                    for i, (body, kind) in enumerate(
                        zip(model.shape_body.numpy(), model.shape_type.numpy(), strict=True)
                    )
                    if body in eligible and kind == newton.GeoType.BOX
                ]
                np.testing.assert_array_equal(model.shape_body.numpy()[ids], geometry["collider_body"])
                np.testing.assert_allclose(
                    model.shape_transform.numpy()[ids], geometry["collider_local_pose"], atol=1e-6
                )
                np.testing.assert_allclose(model.shape_scale.numpy()[ids], geometry["collider_half_size"], atol=1e-6)
            for meta, snap, _ in data:
                if model.particle_count != meta["particle_count"] or snap["body_pose"].shape[1] != model.body_count:
                    raise ValueError("Replay geometry does not match recorded bodies/particles")
                if "body_labels" in meta and list(model.body_label) != meta["body_labels"]:
                    raise ValueError("Replay body ordering does not match recorded bodies")
            if native_rtx:
                # Use the task's authored USD materials, lights, particle widths and camera.
                visualizer = env.unwrapped.sim.visualizers[0]
                state = NewtonManager.get_state_0()
            else:
                state = model.state()
                viewer = ViewerGL(width=panel_width, height=height, headless=True)
                viewer.set_model(model)
                viewer.show_particles = False
                viewer.show_collision = False
                viewer.show_visual = True
                viewer.set_camera(wp.vec3(3.5, -4, 2.4))
                viewer.camera.look_at((0, 0, 0.15))
                particle_colors = wp.full(
                    model.particle_count, wp.vec3(0.32, 0.24, 0.21), dtype=wp.vec3, device=model.device
                )
            frames = max(len(d[1]["time_s"]) for d in data)
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
                    if k >= len(snap["time_s"]):
                        draw.text(
                            (i * panel_width + 20, top + 40),
                            (
                                f"{labels[i]} ended at {snap['time_s'][-1]:.2f} s "
                                f"({meta.get('outcome', 'end of recording')})"
                            ),
                            font=font,
                            fill=colors[i],
                        )
                        continue
                    t = float(snap["time_s"][k])
                    state.body_q.assign(snap["body_pose"][k])
                    state.particle_q.assign(snap["particle_position"][k])
                    if native_rtx:
                        NewtonManager._mark_transforms_changed()
                        NewtonManager._mark_particles_dirty()
                        env.unwrapped.scene.update(0.04)
                        visualizer._camera_tracking_step = -1
                        visualizer.update_camera_tracking()
                        visualizer._render_frame()
                        pixels = visualizer._viewer.get_frame().numpy()
                    else:
                        viewer.begin_frame(t)
                        viewer.log_state(state)
                        viewer.log_points("sand", state.particle_q, radii=0.010, colors=particle_colors)
                        viewer.end_frame()
                        pixels = viewer.get_frame().numpy()
                    frame.paste(Image.fromarray(pixels).convert("RGB"), (i * panel_width, top))
                    draw.text((i * panel_width + 14, 12), labels[i], font=font, fill=colors[i])
                    draw.text(
                        (i * panel_width + 14, 35),
                        f"{'Recorded' if args.particle_evaluation else 'Dynamic'} G1 | 3 x 3 m sand | t = {t:.2f} s",
                        font=font,
                        fill="#23383e",
                    )
                    if args.particle_evaluation:
                        status = "Identical recorded robot poses | particle prediction at 200 Hz | no force feedback"
                    else:
                        outcome = meta["outcome"].replace("_", " ")
                        status = f"Recorded run: {meta['duration_s']:.2f} s ({outcome})"
                    draw.text((i * panel_width + 14, 59), status, font=small_font, fill="#566970")
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
