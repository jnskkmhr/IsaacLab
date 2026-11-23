# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read complete-trajectory splits and the actual MPM collider input states."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .network import material_features


class Episodes:
    """Sample short recurrent windows without changing trajectory split membership."""

    def __init__(self, root, split, *, collider_input="material_step"):
        if collider_input not in ("material_step", "pre_rigid_step"):
            raise ValueError(collider_input)
        self.collider_input = collider_input
        self.records = []
        for path in sorted(Path(root).rglob("manifest.json")):
            manifest = json.loads(path.read_text())
            if manifest.get("schema") != "particle-nerd-object-trajectories-v1":
                continue
            for record in manifest["episodes"]:
                if record["split"] == split:
                    file = path.parent / (record["id"] + ".h5")
                    if not file.is_file():
                        raise FileNotFoundError(file)
                    self.records.append((record, manifest, file))
        if not self.records:
            raise ValueError(f"No {split} trajectories found in {root}")
        ids = [r[0]["id"] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate trajectory IDs in the dataset")

    def num_frames(self, index) -> int:
        with h5py.File(self.records[index][2], "r") as handle:
            return int(len(handle["particle_position"]))

    def sequence(self, rng, horizon, device, *, index=None, start=None):
        record, manifest, path = self.records[int(rng.integers(len(self.records))) if index is None else index]
        with h5py.File(path, "r") as handle:
            if not handle.attrs.get("complete"):
                raise ValueError(f"Incomplete trajectory: {path}")
            final = len(handle["particle_position"]) - horizon - 1
            if start is None:
                start = int(rng.integers(final + 1))
            if start < 0 or start > final:
                raise ValueError("Requested window exceeds this trajectory")
            lower = max(0, start - 2)
            positions = handle["particle_position"][lower : start + horizon + 1]
            if lower == 0 and start < 2:
                if record.get("motion") == "throw":
                    # The block is released with a known velocity: encode it in the history instead of
                    # describing a block at rest.
                    from .protocol import release_velocity

                    velocity = release_velocity(record)[:3]
                    step = float(record.get("dt_s", 0.005))
                    padding = np.stack(
                        [positions[0] - m * step * velocity[None, :] for m in range(2 - start, 0, -1)]
                    ).astype(positions.dtype)
                else:
                    padding = np.repeat(positions[:1], 2 - start, axis=0)
                positions = np.concatenate((padding, positions))
            body = handle["collider_state"][start : start + horizon]
            if self.collider_input == "pre_rigid_step" and record["mode"] == "dynamic":
                # The current G1 neural input is captured before rigid integration.
                # The recorded collider_state instead contains rewound free velocity
                # used by force reconstruction. Keep these two inputs distinct.
                body = handle["body_state"][start : start + horizon].copy()
                release_step = round(1.0 / manifest["dt_s"])
                if start <= release_step < start + horizon:
                    from .protocol import release_velocity

                    body[release_step - start, 7:] = release_velocity(record)
            labels = handle["shape_labels"].asstr()[:]
            mu = handle["shape_material_mu"][:]
            shape_body = handle["shape_body"][:]
            moving_mu = mu[shape_body >= 0]
            if len(moving_mu) != 1:
                raise ValueError("Expected one object collision shape")
            friction = [float(moving_mu[0])]
            for wall in manifest["boundary_boxes"]:
                name = {
                    "floor": "Floor",
                    "wall_x0": "WallX0",
                    "wall_x1": "WallX1",
                    "wall_y0": "WallY0",
                    "wall_y1": "WallY1",
                }[wall["name"]]
                indices = [i for i, label in enumerate(labels) if f"/{name}/" in label]
                if len(indices) != 1:
                    raise ValueError(f"Expected one MPM boundary shape for {name}: {indices}")
                friction.append(float(mu[indices[0]]))
            material = material_features(
                manifest, float(handle["particle_mass"][0]), float(handle["particle_radius"][0]), manifest["dt_s"]
            )

        def tensor(value):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)

        boundary_pose = tensor([w["pose"] for w in manifest["boundary_boxes"]])
        poses = torch.cat((tensor(body[:, None, :7]), boundary_pose[None].expand(horizon, -1, -1)), dim=1)
        velocities = torch.cat((tensor(body[:, None, 7:]), torch.zeros((horizon, 5, 6), device=device)), dim=1)
        kinds = torch.tensor(
            [{"box": 1, "sphere": 2, "cylinder": 4, "freefall": 1, "rest": 1}[record["shape"]]],
            device=device,
            dtype=torch.int32,
        )
        kinds = torch.cat((kinds, torch.ones(5, device=device, dtype=torch.int32)))
        sizes = tensor([manifest["body_size_m"], *[w["size_m"] for w in manifest["boundary_boxes"]]]) / 2
        return {
            "history": tensor(positions[:3].transpose(1, 0, 2)).contiguous(),
            "targets": tensor(positions[3:]),
            "poses": poses.contiguous(),
            "velocities": velocities.contiguous(),
            "material": tensor(material),
            "kinds": kinds,
            "sizes": sizes.contiguous(),
            "friction": tensor(friction),
            "dt": manifest["dt_s"],
            "id": record["id"],
            "start": start,
            "shape": record["shape"],
            "mode": record["mode"],
            "collider_input": self.collider_input,
        }
