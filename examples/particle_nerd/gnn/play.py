# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""View MPM-free GNN particles with recorded rigid motion for a held-out episode.

This isolates particle-model robustness. The box follows the recorded trajectory;
interactive body-force coupling is not claimed by this playback entry point.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import newton
import numpy as np
import torch
import warp as wp

from particle_nerd.gnn.inference import ParticlePredictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--selector", type=Path, help="Optional learned particle-activity checkpoint")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument(
        "--compile",
        action="store_true",
        help="Compile the Torch network; first step includes compilation",
    )
    p.add_argument(
        "--reference",
        action="store_true",
        help="Render MPM reference particles instead for comparison",
    )
    args = p.parse_args()
    with h5py.File(args.episode, "r") as f:
        m = json.loads(f.attrs["manifest"])
        frames = f["frame_steps"][:]
        start = int(np.searchsorted(frames, m["settled_step"]))
        stop = min(start + args.steps, len(frames) - 1)
        predictor_type = ParticlePredictor
        selector_args = {}
        if args.selector:
            from particle_nerd.gnn.region_selection import SelectedPredictor

            predictor_type = SelectedPredictor
            selector_args = dict(selector_checkpoint=args.selector, mode="learned")
        predictor = predictor_type(
            args.checkpoint,
            f["particle_position"][start],
            f["particle_velocity"][start],
            m,
            float(f["particle_mass"][0]),
            float(f["particle_radius"][0]),
            compile_model=args.compile,
            **selector_args,
        )
        predictor.history.copy_(
            torch.as_tensor(
                np.concatenate(
                    [
                        f["particle_velocity"][max(0, start - k)]
                        for k in range(predictor.model.config["history"] - 1, -1, -1)
                    ],
                    axis=-1,
                ),
                device="cuda",
            )
        )
        poses = torch.as_tensor(f["body_pose"][:], device="cuda")
        velocities = torch.as_tensor(f["body_velocity"][:], device="cuda")
        viewer = None
        if not args.headless:
            from newton.viewer import ViewerGL

            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            q = f["body_pose"][frames[start]]
            body = builder.add_body(xform=wp.transform(q[:3], q[3:]), label="reference_box_motion")
            builder.add_shape_box(body, hx=0.06, hy=0.06, hz=0.04, color=(0.32, 0.38, 0.42))
            model = builder.finalize(device="cuda:0")
            state = model.state()
            viewer = ViewerGL(width=1200, height=800)
            viewer.set_model(model)
            viewer.set_camera(wp.vec3(1.8, -2.4, 1.8))
            viewer.camera.look_at((0.0, 0.0, 0.18))
            color = wp.array(
                np.tile([0.32, 0.24, 0.21], (len(predictor.x), 1)).astype("f4"),
                dtype=wp.vec3,
                device="cuda:0",
            )
        print(
            "Particle dynamics:",
            "MPM recording" if args.reference else "GNN; no MPM solver",
            flush=True,
        )
        print(
            "Box motion: replayed held-out recording; no closed-loop rigid force prediction",
            flush=True,
        )
        if args.selector and not args.reference:
            print(
                "Learned activity selection enabled; inactive particles retain their positions",
                flush=True,
            )
        started = time.perf_counter()
        for frame in range(start, stop):
            if viewer is not None and not viewer.is_running():
                break
            if args.reference:
                predictor.x.copy_(torch.as_tensor(f["particle_position"][frame + 1], device="cuda"))
            else:
                predictor.step(poses[frames[frame]], velocities[frames[frame]])
            if viewer is not None:
                wp.copy(
                    state.body_q,
                    wp.from_torch(
                        poses[frames[frame + 1] : frames[frame + 1] + 1],
                        dtype=wp.transform,
                    ),
                )
                viewer.begin_frame((frame - start + 1) * 0.02)
                viewer.log_state(state)
                viewer.log_points(
                    "sand",
                    wp.from_torch(predictor.x, dtype=wp.vec3),
                    radii=0.012,
                    colors=color,
                )
                viewer.end_frame()
                remaining = (frame - start + 1) * 0.02 - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
            # Diagnostics intentionally synchronize outside predictor.step.
            # Stop before an invalid state reaches the next neighborhood search.
            check = predictor.diagnostics()
            if (frame - start) % 25 == 0 or frame == stop - 1:
                print(json.dumps(dict(frame=frame - start + 1, **check)), flush=True)
            if check["graph_overflow"] or not check["finite"]:
                raise RuntimeError("Invalid learned rollout; inspect diagnostics")
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
