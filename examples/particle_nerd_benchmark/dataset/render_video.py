# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render dataset episodes to MP4 with the Newton OpenGL viewer. Pure replay of the stored states, no physics."""

import argparse
import subprocess
from pathlib import Path

import h5py
import pyglet

pyglet.options["headless"] = True  # no X display: render offscreen with EGL, as the Isaac Lab visualizer does

import newton  # noqa: E402
import numpy as np
import warp as wp

from .episodes import CUBE_SIZE, DT, open_ground_boundary_boxes, tray_boundary_boxes

SAND_COLOR = (0.72, 0.59, 0.36)
OBJECT_COLOR = (0.15, 0.40, 0.75)
OBJECT_SIZES = {
    "cube": (CUBE_SIZE,) * 3,
    "box": (0.24, 0.18, 0.12),
    "sphere": (0.2, 0.2, 0.2),
    "cylinder": (0.2, 0.2, 0.12),
}
FIELDS = {"freefall": (0.0, 3.0, 0.0), "rest_bed": (0.0, 0.0, -0.1)}


def build_model(shape: str, category: str, bed_size, device: str, has_body: bool):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
    boxes = open_ground_boundary_boxes() if category == "freefall" else tray_boundary_boxes(bed_size)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0, margin=0.0)
    for box in boxes:
        if box["name"] != "floor" and (category == "freefall" or box["name"] in ("wall_y0", "wall_x1")):
            continue  # open-ground walls are 10 m away; the two near walls would hide the bed
        half = [s / 2.0 for s in box["size_m"]]
        pose = box["pose"]
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])),
            hx=half[0],
            hy=half[1],
            hz=half[2],
            cfg=cfg,
            color=(0.55, 0.58, 0.60),
        )
    if has_body and shape in OBJECT_SIZES:
        body = builder.add_body(xform=wp.transform_identity())
        sx, sy, sz = OBJECT_SIZES[shape]
        if shape == "sphere":
            builder.add_shape_sphere(body, radius=0.5 * sx, color=OBJECT_COLOR)
        elif shape == "cylinder":
            builder.add_shape_cylinder(body, radius=0.5 * sx, half_height=0.5 * sz, color=OBJECT_COLOR)
        else:
            builder.add_shape_box(body, hx=0.5 * sx, hy=0.5 * sy, hz=0.5 * sz, color=OBJECT_COLOR)
    return builder.finalize(device=device)


def render(path: Path, out: Path, width=960, height=540, stride=7, still: float | None = None, device="cuda:0"):
    with h5py.File(path, "r") as f:
        category, shape = str(f.attrs["category"]), str(f.attrs["shape"])
        spacing = float(f.attrs["particle_spacing_m"])
        bed = json_list(f.attrs.get("bed_size_m"))
        steps = int(f.attrs["num_steps"])
        pos = f["particles/pos"]
        pose = f["body/pose"][:] if "body" in f else None
        frames = [round(still / DT)] if still is not None else list(range(0, steps + 1, stride))
        model = build_model(shape, category, bed or (2.0, 1.2, 0.3), device, pose is not None)
        viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
        viewer.set_model(model)
        eye = np.array({"freefall": (1.0, -1.3, 0.25), "rest_bed": (1.8, -2.2, 0.9)}.get(category, (1.5, -1.8, 0.75)))
        look = np.array({"freefall": (0.0, 0.0, -0.15)}.get(category, (0.0, 0.0, 0.0)))
        d = look - eye
        viewer.set_camera(
            wp.vec3(*eye),
            pitch=float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))),
            yaw=float(np.degrees(np.arctan2(d[1], d[0]))),
        )
        state = model.state()
        radii = wp.full(pos.shape[1], 0.5 * spacing, dtype=wp.float32, device=device)
        colors = wp.full(pos.shape[1], wp.vec3(*SAND_COLOR), dtype=wp.vec3, device=device)
        encoder = None
        if still is None:
            encoder = subprocess.Popen(
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
                    f"{width}x{height}",
                    "-r",
                    f"{1.0 / (stride * DT):.3f}",
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "23",
                    str(out),
                ],
                stdin=subprocess.PIPE,
            )
        image = None
        for index in frames:
            if pose is not None:
                state.body_q.assign(pose[index][None].astype(np.float32))
            viewer.begin_frame(index * DT)
            viewer.log_state(state)
            viewer.log_points("/sand", wp.array(pos[index], dtype=wp.vec3, device=device), radii, colors)
            viewer.end_frame()
            image = viewer.get_frame().numpy()
            if encoder is not None:
                encoder.stdin.write(np.ascontiguousarray(image).tobytes())
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()
        else:
            import matplotlib.image as mpimg

            mpimg.imsave(out, image)
        viewer.close()


def json_list(value):
    import json

    return None if value is None else json.loads(value) if isinstance(value, str) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--still", type=float, help="Render a single PNG at this time [s] instead of a video")
    parser.add_argument("--stride", type=int, default=7, help="Frames between rendered states (7 = about real time)")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    render(args.episode, args.out, stride=args.stride, still=args.still)


if __name__ == "__main__":
    main()
