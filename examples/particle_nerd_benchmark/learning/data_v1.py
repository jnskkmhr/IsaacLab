# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the terrain / category dataset (one HDF5 file per episode) as recurrent windows for training and evaluation.

The interface matches `data.Episodes` (`records`, `sequence`), so the training and evaluation code can use either.
Differences from the first dataset format that matter for the neural input:

- The collider input is the recorded body state. Kinematic objects use the state after the step (it equals the
  command that the MPM step saw). Dynamic objects use the state before the step, with the release velocity restored
  at the release step. The old format stored the MPM collider state with the lagged sand feedback removed, which the
  new files do not contain.
- Sand-only episodes (`rest_bed`, `freefall`) have no body: a far-field placeholder box fills the collider slot.
- Free-fall blocks land on the terrain's bed. Their particles are the bed's followed by the block's.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ..dataset.episodes import tray_boundary_boxes
from .network import material_features

OBJECT_CATEGORIES = ("lateral_push", "vertical_push", "impact", "rest")
SAND_CATEGORIES = ("freefall", "rest_bed")
DEFAULT_CATEGORIES = (*OBJECT_CATEGORIES, *SAND_CATEGORIES)
KIND = {"box": 1, "cube": 1, "sphere": 2, "cylinder": 4, "rest": 1, "sandblock": 1}
FAR_POSE = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
FAR_SIZE = (0.24, 0.18, 0.12)
CONTACT_START = {  # first step of a window that contains the interaction, per category / motion
    "lateral_push": 200,
    "vertical_push": 400,
    "impact": 200,
    "rest": 20,
    "freefall": 40,
    "rest_bed": 1000,
}
REST_START_S = {"rest": 0.5, "rest_bed": 0.0}
"""Earliest start [s] of a long resting-bed unroll."""
HOVER_STEPS = 200


class EpisodesV1:
    """Sample recurrent windows from the files of one split, optionally restricted by category and terrain."""

    def __init__(self, root, split, *, categories=DEFAULT_CATEGORIES, terrains=None, collider_input="pre_step"):
        self.collider_input = collider_input
        self.records = []
        for path in sorted(Path(root).rglob("*.hdf5")):
            with h5py.File(path, "r") as handle:
                a = handle.attrs
                if not a.get("complete", False) or str(a.get("schema_version", "")) != "particle-nerd-dataset-v1":
                    continue
                if str(a["split"]) != split or str(a["category"]) not in categories:
                    continue
                if terrains is not None and str(a["terrain"]) not in terrains:
                    continue
                params = json.loads(a["params"])
                category, material = str(a["category"]), json.loads(a["material"])
                size = json.loads(a["bed_size_m"]) if "bed_size_m" in a and str(a["bed_size_m"]) != "null" else None
                motion_type = str(a["motion_type"])
                num_block = int(a["num_block_particles"]) if "num_block_particles" in a else 0
                spacing = float(a["particle_spacing_m"])
                mass = float(a["particle_mass_kg"])
                object_size = (
                    json.loads(a["object_size_m"])
                    if "object_size_m" in a and str(a["object_size_m"]) != "null"
                    else list(FAR_SIZE)
                )
            shape = {"rest_bed": "rest", "freefall": "sandblock"}.get(category, params["shape"])
            record = {
                **params,
                "id": params["id"],
                "shape": shape,
                "mode": "prescribed" if motion_type == "kinematic" else "dynamic",
                "motion": params["motion"],
                "dt_s": float(params["dt_s"]),
                "rest_unroll": category in ("rest", "rest_bed"),
                "rest_start_s": REST_START_S.get(category, 0.0),
                "contact_start": CONTACT_START[category],
                "num_block_particles": num_block,
                "particle_mass": mass,
                "particle_radius": spacing / 2.0,
            }
            manifest = {
                "material": material,
                "boundary_boxes": tray_boundary_boxes(size or (2.0, 1.2, 0.30)),
                "body_size_m": object_size,
                "dt_s": float(params["dt_s"]),
            }
            self.records.append((record, manifest, path))
        if not self.records:
            raise ValueError(f"No {split} trajectories found in {root} for categories {categories}")
        ids = [r[0]["id"] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate trajectory IDs in the dataset")

    def num_frames(self, index) -> int:
        with h5py.File(self.records[index][2], "r") as handle:
            return int(handle["particles/pos"].shape[0])

    def sequence(self, rng, horizon, device, *, index=None, start=None):
        record, manifest, path = self.records[int(rng.integers(len(self.records))) if index is None else index]
        dt = record["dt_s"]
        category = record["category"]
        with h5py.File(path, "r") as handle:
            frames = handle["particles/pos"].shape[0]
            final = frames - horizon - 1
            if start is None:
                start = int(rng.integers(final + 1))
            if start < 0 or start > final:
                raise ValueError("Requested window exceeds this trajectory")
            lower = max(0, start - 2)
            positions = handle["particles/pos"][lower : start + horizon + 1]
            if lower == 0 and start < 2:
                if category == "freefall" and record["motion"] == "toss":
                    # The block is released with a known velocity: encode it in the history (the bed stays at rest).
                    velocity = np.zeros((positions.shape[1], 3), dtype=np.float32)
                    velocity[positions.shape[1] - record["num_block_particles"] :] = np.asarray(
                        record["release_velocity"][:3], dtype=np.float32
                    )
                    padding = np.stack([positions[0] - m * dt * velocity for m in range(2 - start, 0, -1)])
                    padding = padding.astype(positions.dtype)
                else:
                    padding = np.repeat(positions[:1], 2 - start, axis=0)
                positions = np.concatenate((padding, positions))
            if "body" in handle:
                if record["mode"] == "prescribed":
                    shift = start + 1  # the command that the MPM step saw is the state recorded after the step
                    pose = handle["body/pose"][shift : shift + horizon]
                    vel = handle["body/vel"][shift : shift + horizon]
                else:
                    pose = handle["body/pose"][start : start + horizon]
                    vel = handle["body/vel"][start : start + horizon].copy()
                    if category == "impact" and start <= HOVER_STEPS < start + horizon:
                        vel[HOVER_STEPS - start] = np.asarray(record["release_velocity"], dtype=np.float32)
                body = np.concatenate((pose, vel), axis=-1)
            else:
                body = np.tile(np.concatenate((FAR_POSE, np.zeros(6, dtype=np.float32)))[None], (horizon, 1))

        def tensor(value):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)

        boxes = manifest["boundary_boxes"]
        boundary_pose = tensor([w["pose"] for w in boxes])
        poses = torch.cat((tensor(body[:, None, :7]), boundary_pose[None].expand(horizon, -1, -1)), dim=1)
        velocities = torch.cat((tensor(body[:, None, 7:]), torch.zeros((horizon, 5, 6), device=device)), dim=1)
        kinds = torch.cat(
            (
                torch.tensor([KIND[record["shape"]]], device=device, dtype=torch.int32),
                torch.ones(5, device=device, dtype=torch.int32),
            )
        )
        sizes = tensor([manifest["body_size_m"], *[w["size_m"] for w in boxes]]) / 2
        material = material_features(manifest, record["particle_mass"], record["particle_radius"], dt)
        return {
            "history": tensor(positions[:3].transpose(1, 0, 2)).contiguous(),
            "targets": tensor(positions[3:]),
            "poses": poses.contiguous(),
            "velocities": velocities.contiguous(),
            "material": tensor(material),
            "kinds": kinds,
            "sizes": sizes.contiguous(),
            "friction": tensor([1.0] * 6),
            "dt": dt,
            "id": record["id"],
            "start": start,
            "shape": record["shape"],
            "mode": record["mode"],
            "collider_input": self.collider_input,
        }
