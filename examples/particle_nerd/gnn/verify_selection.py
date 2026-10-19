# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify inactive-particle invariance and detection of transient reference motion."""

import argparse
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp

from particle_nerd.gnn.features import Features
from particle_nerd.gnn.region_selection import (
    MotionSelector,
    freeze_inactive,
    reference_activity,
    selection_inputs,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    wp.init()
    old = torch.tensor([[0.0, 0.0, 0.1], [0.04, 0.0, 0.1], [0.2, 0.0, 0.1]], device="cuda")
    x = old + 0.05
    v = torch.ones_like(x)
    history = torch.ones((3, 9), device="cuda")
    active = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
    overflow = wp.zeros(1, dtype=int, device="cuda")
    before = x.clone()
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        wp.launch(
            freeze_inactive,
            3,
            [
                wp.from_torch(x, dtype=wp.vec3),
                wp.from_torch(v, dtype=wp.vec3),
                wp.from_torch(history),
                wp.from_torch(old, dtype=wp.vec3),
                wp.from_torch(active),
                overflow,
            ],
        )
    torch.testing.assert_close(x[active == 0], old[active == 0], atol=0, rtol=0)
    torch.testing.assert_close(x[active == 1], before[active == 1], atol=0, rtol=0)
    assert (v[active == 0] == 0).all() and (history[active == 0, -3:] == 0).all()
    assert (history[:, :6] == 1).all()
    ftr = Features(3, capacity_per_particle=4)
    inputs = ftr.prepare(
        old,
        history,
        torch.tensor([2.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], device="cuda"),
        torch.zeros(6, device="cuda"),
        torch.ones(13, device="cuda"),
        7.5,
    )
    selector = MotionSelector().cuda()
    saved_nodes = inputs[0].clone()
    saved_boundary = inputs[2].clone()
    features = selection_inputs(inputs)
    torch.testing.assert_close(inputs[0], saved_nodes, atol=0, rtol=0)
    torch.testing.assert_close(inputs[2], saved_boundary, atol=0, rtol=0)
    assert features.shape == (3, 48)
    assert torch.isfinite(selector(features)).all()
    with tempfile.TemporaryDirectory() as td:
        with h5py.File(Path(td) / "motion.hdf5", "w") as f:
            f.create_dataset("frame_steps", data=np.arange(6))
            positions = np.zeros((6, 3, 3), dtype="f4")
            positions[1, 1, 0] = 0.001
            positions[:, 2, 0] = np.arange(6) * 0.0002
            f.create_dataset("particle_position", data=positions)
            assert reference_activity(f, 0).tolist() == [False, True, True]
    result = dict(
        inactive_positions_exactly_preserved=True,
        inactive_endpoint_velocity_zero=True,
        active_state_unchanged_by_freezing=True,
        transient_motion_labels_detected=True,
        selector_input_features=48,
        shared_motion_features_preserved=True,
        device=torch.cuda.get_device_name(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
