# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure unintended movement of reference-stationary particles in learned rollouts."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for path in sorted(args.references.glob("episode-*-attempt-0.hdf5")):
        index = int(path.name.split("-")[1])
        prediction = args.predictions / f"rollout-{index:05d}.hdf5"
        if not prediction.exists():
            continue
        with h5py.File(path) as ref, h5py.File(prediction) as pred:
            start = int(pred.attrs["start_frame"])
            count = int(pred.attrs["valid_frames"])
            initial = ref["particle_position"][start]
            reference_displacement = np.zeros(len(initial))
            predicted_displacement = np.zeros(len(initial))
            for i in range(count):
                reference_displacement = np.maximum(
                    reference_displacement,
                    np.linalg.norm(ref["particle_position"][start + i] - initial, axis=1),
                )
                predicted_displacement = np.maximum(
                    predicted_displacement,
                    np.linalg.norm(pred["particle_position"][i] - initial, axis=1),
                )
            quiet = reference_displacement < 0.001
            unexpected = quiet & (predicted_displacement > 0.01)
            rows.append(
                dict(
                    index=index,
                    duration_s=(count - 1) * 0.02,
                    reference_stationary_particles=int(quiet.sum()),
                    stationary_particles_predicted_to_move_over_1cm=int(unexpected.sum()),
                    fraction_of_reference_stationary_particles_moving_over_1cm=float(unexpected.sum() / quiet.sum())
                    if quiet.any()
                    else None,
                    reference_final_max_height_m=float(ref["particle_position"][start + count - 1, :, 2].max()),
                    predicted_final_max_height_m=float(pred["particle_position"][count - 1, :, 2].max()),
                )
            )
    output = dict(
        definition=(
            "Reference-stationary means displacement from the initial settled position "
            "remains below 1 mm throughout the evaluated interval. Unintended movement "
            "means the corresponding GNN particle moves more than 1 cm at any time. "
            "Heights are particle-center z above the rigid-ground plane."
        ),
        episodes=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
