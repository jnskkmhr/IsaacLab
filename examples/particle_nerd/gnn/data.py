# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded-memory HDF5 training samples with episode-level train/test separation."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from particle_nerd.gnn.features import material_features


class Episodes:
    def __init__(self, root, split, history=3):
        self.root = Path(root)
        self.history = history
        self.records = []
        for path in sorted(self.root.rglob("*.campaign.json")):
            record = json.loads(path.read_text())
            if record["split"] == split:
                record["path"] = path.parent / record["file"]
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No {split} episodes under {root}")
        ids = [r["index"] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate episode IDs; keep one accepted sample per episode")

    def sample(self, rng, device="cuda:0", episode=None, frame=None):
        record = self.records[int(rng.integers(len(self.records))) if episode is None else episode]
        with h5py.File(record["path"], "r") as f:
            m = json.loads(f.attrs["manifest"])
            steps = f["frame_steps"][:]
            first = int(np.searchsorted(steps, m["settled_step"]))
            if frame is None:
                # Include falling/settling, while most samples cover the approved
                # five-second disturbance window rather than only initial impact.
                lower = first if rng.random() < 0.8 else 0
                frame = int(rng.integers(lower, len(steps) - 1))

            def tensor(a):
                return torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)

            hist = np.concatenate(
                [f["particle_velocity"][max(0, frame - k)] for k in range(self.history - 1, -1, -1)],
                axis=-1,
            )
            return dict(
                x=tensor(f["particle_position"][frame]),
                history=tensor(hist),
                next_x=tensor(f["particle_position"][frame + 1]),
                next_v=tensor(f["particle_velocity"][frame + 1]),
                pose=tensor(f["body_pose"][steps[frame]]),
                body_velocity=tensor(f["body_velocity"][steps[frame]]),
                material=tensor(material_features(m, float(f["particle_mass"][0]), float(f["particle_radius"][0]))),
                mass=m["body_mass_kg"],
                index=record["index"],
                frame=frame,
                manifest=m,
            )
