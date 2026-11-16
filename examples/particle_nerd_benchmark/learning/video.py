# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render paired MPM/NeRD saved states at 50 fps without running a physics solver."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import newton
import newton.viewer
import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    root = args.evaluation
    meta = json.loads((root / "evaluation.json").read_text())
    reference_path = Path(meta["reference_path"])
    if file_hash(reference_path) != meta["reference_sha256"]:
        raise ValueError("Reference hash differs from the plotted evaluation")
    manifest = json.loads((reference_path.parent / "manifest.json").read_text())
    wp.init()
    with h5py.File(reference_path, "r") as reference, h5py.File(root / "prediction.h5", "r") as prediction:
        builder = newton.ModelBuilder()
        shape = meta["episode"]["shape"]
        if shape != "freefall":
            body = builder.add_body(mass=1.0, is_kinematic=True, label="InteractingObject")
            size = np.asarray(manifest["body_size_m"]) / 2
            color = (0.12, 0.32, 0.25)
            if shape == "box":
                builder.add_shape_box(body, hx=float(size[0]), hy=float(size[1]), hz=float(size[2]), color=color)
            elif shape == "sphere":
                builder.add_shape_sphere(body, radius=float(size[0]), color=color)
            else:
                builder.add_shape_cylinder(body, radius=float(size[0]), half_height=float(size[2]), color=color)
        for wall in manifest["boundary_boxes"]:
            if shape == "freefall" and wall["name"] != "floor":
                continue
            half = np.asarray(wall["size_m"]) / 2
            builder.add_shape_box(
                -1,
                xform=wp.transform(wall["pose"][:3], wall["pose"][3:]),
                hx=float(half[0]),
                hy=float(half[1]),
                hz=float(half[2]),
                color=(0.25, 0.30, 0.33),
                label=wall["name"],
            )
        x = reference["particle_position"][0]
        builder.add_particles(
            pos=x, vel=np.zeros_like(x), mass=reference["particle_mass"][:], radius=reference["particle_radius"][:]
        )
        model = builder.finalize(device="cuda:0")
        state = model.state()
        width, height, header = 800, 600, 36
        viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True, vsync=False)
        viewer.set_model(model)
        viewer.show_particles = True
        viewer.set_camera(wp.vec3(1.65, -2.0, 1.55))
        viewer.camera.look_at((0.0, 0.0, -0.05))
        viewer.camera.fov = 50.0
        colors = wp.full(len(x), value=wp.vec3(0.72, 0.59, 0.36), dtype=wp.vec3, device="cuda:0")
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
        frame_indices = list(range(4, len(prediction["particle_position"]), 4))
        video_path = root / "comparison.mp4"
        encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{2 * width}x{height + header}",
                "-r",
                "50",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            stdin=subprocess.PIPE,
        )
        pred_label = os.environ.get("BENCHMARK_MODEL_LABEL") or (
            "Particle World"
            if "cconv_warp.py" in meta.get("network_source_sha256", {})
            else ("Particle NeRD-Local" if len(meta.get("network_source_sha256", {})) == 2 else "Particle NeRD")
        )
        try:
            for frame in frame_indices:
                canvas = Image.new("RGB", (2 * width, height + header), "white")
                draw = ImageDraw.Draw(canvas)
                for column, (source, label) in enumerate(((reference, "Implicit MPM"), (prediction, pred_label))):
                    state.particle_q.assign(source["particle_position"][frame])
                    if shape != "freefall":
                        state.body_q.assign(source["body_state"][frame, :7][None])
                    viewer.begin_frame(frame * meta["native_dt_s"])
                    viewer.log_state(state)
                    viewer.log_points("/model/particles", state.particle_q, radii=model.particle_radius, colors=colors)
                    viewer.end_frame()
                    pixels = viewer.get_frame(render_ui=False).numpy()
                    canvas.paste(Image.fromarray(pixels), (column * width, header))
                    draw.text((column * width + width / 2, header / 2), label, font=font, fill="#1c2a32", anchor="mm")
                if frame == frame_indices[0]:
                    canvas.save(root / "comparison_poster.png")
                encoder.stdin.write(np.asarray(canvas).tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Video encoder failed")
        finally:
            if encoder.poll() is None:
                encoder.stdin.close()
                encoder.wait()
            viewer.close()
        (root / "video_provenance.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": meta["checkpoint_sha256"],
                    "reference_sha256": meta["reference_sha256"],
                    "prediction_sha256": file_hash(root / "prediction.h5"),
                    "video_sha256": file_hash(video_path),
                    "fps": 50,
                    "native_dt_s": meta["native_dt_s"],
                    "state_indices": frame_indices,
                    "rendering": "Saved recorded and predicted states; no physics solver is stepped during rendering",
                    "camera_eye": [1.65, -2.0, 1.55],
                    "camera_target": [0.0, 0.0, -0.05],
                    "fov_degrees": 50.0,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
