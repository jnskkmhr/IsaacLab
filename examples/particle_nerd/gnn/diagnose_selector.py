# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate fixed activity selection on reference states of preselected test recordings.

This is a post-test diagnosis, not a checkpoint-selection or threshold-tuning step.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from particle_nerd.gnn.features import Features, material_features
from particle_nerd.gnn.region_selection import MotionSelector, reference_activity, selection_inputs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--selector", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    saved = torch.load(args.selector, weights_only=False, map_location="cuda")
    model = MotionSelector(**saved["config"]).cuda().eval()
    model.load_state_dict(saved["model"])
    results = []
    for path in sorted(args.reference.glob("episode-*.hdf5")):
        with h5py.File(path) as f:
            manifest = json.loads(f.attrs["manifest"])
            steps = f["frame_steps"][:]
            start = int(np.searchsorted(steps, manifest["settled_step"]))
            features = Features(f["particle_position"].shape[1])
            material = torch.as_tensor(
                material_features(
                    manifest,
                    float(f["particle_mass"][0]),
                    float(f["particle_radius"][0]),
                ),
                device="cuda",
            )
            counts = np.zeros(4, dtype=np.int64)
            for frame in range(start, len(steps) - 1):

                def tensor(x):
                    return torch.as_tensor(x, device="cuda", dtype=torch.float32)

                inputs = features.prepare(
                    tensor(f["particle_position"][frame]),
                    tensor(
                        np.concatenate(
                            [f["particle_velocity"][max(0, frame - k)] for k in (2, 1, 0)],
                            axis=-1,
                        )
                    ),
                    tensor(f["body_pose"][steps[frame]]),
                    tensor(f["body_velocity"][steps[frame]]),
                    material,
                    manifest["body_mass_kg"],
                    compact=True,
                )
                predicted = (model(selection_inputs(inputs)).sigmoid() >= saved["threshold"]).cpu().numpy()
                truth = reference_activity(f, frame, saved["label_horizon"], saved["label_distance"])
                counts += [
                    np.sum(predicted & truth),
                    np.sum(predicted & ~truth),
                    np.sum(~predicted & truth),
                    np.sum(~predicted & ~truth),
                ]
            tp, fp, fn, tn = counts
            row = dict(
                index=int(path.name.split("-")[1]),
                frames=len(steps) - start - 1,
                precision=float(tp / max(1, tp + fp)),
                recall=float(tp / max(1, tp + fn)),
                false_positive_rate=float(fp / max(1, fp + tn)),
                counts=counts.tolist(),
            )
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                input="Reference MPM states at every step; fixed selected model and threshold",
                diagnostic_only=True,
                trajectories=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
