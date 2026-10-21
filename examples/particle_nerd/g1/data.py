# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded-memory physics-rate G1 records and collider geometry transforms."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from particle_nerd.gnn.features import material_features


def rotate(q, v):
    cross = 2 * torch.linalg.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * cross + torch.linalg.cross(q[..., :3], cross, dim=-1)


def multiply(a, b):
    return torch.cat(
        (
            a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.linalg.cross(a[..., :3], b[..., :3], dim=-1),
            a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True),
        ),
        -1,
    )


def collider_state(body_pose, body_velocity, geometry):
    """Compose actual collision boxes; convert COM velocity to shape-origin velocity."""
    ids = geometry["body_ids"]
    b, velocity = body_pose[ids], body_velocity[ids]
    local = geometry["local_pose"]
    offset = rotate(b[:, 3:], local[:, :3])
    position = b[:, :3] + offset
    rotation = multiply(b[:, 3:], local[:, 3:])
    com_offset = rotate(b[:, 3:], geometry["com"][ids])
    linear = velocity[:, :3] + torch.linalg.cross(velocity[:, 3:], offset - com_offset, dim=-1)
    return torch.cat((position, rotation), -1), torch.cat((linear, velocity[:, 3:]), -1)


def read_geometry(f, device):
    def tensor(a):
        return torch.as_tensor(a, device=device, dtype=torch.float32)

    return dict(
        body_ids=torch.as_tensor(f["collider_body"][:], device=device, dtype=torch.long),
        local_pose=tensor(f["collider_local_pose"][:]),
        sizes=tensor(f["collider_half_size"][:]),
        friction=tensor(f["collider_friction"][:]),
        com=tensor(f["body_com_local"][:]),
        kinds=torch.ones(len(f["collider_body"]), dtype=torch.int32, device=device),
    )


class Episodes:
    def __init__(self, root, split, history=3, stride=1):
        self.history, self.stride = history, stride
        self.records = []
        for path in sorted(Path(root).rglob("episode-*.hdf5")):
            if ".partial." in path.name:
                continue
            with h5py.File(path, "r") as f:
                m = json.loads(f.attrs["manifest"])
                if m["split"] == split:
                    times = f["time_s"][:]
                    if len(times) > history and not np.allclose(np.diff(times), 0.005, atol=1e-8):
                        raise ValueError(f"Nonuniform physics records: {path}")
                    self.records.append(dict(path=path, index=m["trial"]["trial_id"], frames=len(times), manifest=m))
        ids = [r["index"] for r in self.records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicated {split} episode IDs")
        self.dt = 0.005 * stride

    def sequence(self, rng, steps, device, episode=None, frame=None):
        eligible = [r for r in self.records if r["frames"] > (self.history - 1 + steps) * self.stride]
        r = eligible[int(rng.integers(len(eligible))) if episode is None else episode % len(eligible)]
        first, end = (self.history - 1) * self.stride, r["frames"] - steps * self.stride
        start = int(rng.integers(first, end)) if frame is None else frame
        if not first <= start < end:
            raise ValueError(f"Invalid sequence frame {start} for {r['path']}")

        def tensor(a):
            return torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)

        with h5py.File(r["path"], "r") as f:
            history = np.stack(
                [f["particle_position"][start - k * self.stride] for k in reversed(range(self.history))], axis=1
            )
            targets = np.stack([f["particle_position"][start + (k + 1) * self.stride] for k in range(steps)])
            indices = start + np.arange(steps) * self.stride
            return dict(
                history=tensor(history),
                targets=tensor(targets),
                poses=tensor(f["body_pose"][indices]),
                velocities=tensor(f["body_velocity"][indices]),
                geometry=read_geometry(f, device),
                material=tensor(
                    material_features(
                        r["manifest"], float(f["particle_mass"][0]), float(f["particle_radius"][0]), self.dt
                    )
                ),
                trial_id=r["index"],
                frame=start,
            )
