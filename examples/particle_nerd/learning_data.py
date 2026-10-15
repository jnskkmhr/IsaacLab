# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode-level train/validation/test boundaries for learned particle motion."""

import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np


class EpisodeStore:
    def __init__(self, directory, split, cache_size=8):
        self.entries = []
        self.cache = OrderedDict()
        self.cache_size = cache_size
        for path in sorted(Path(directory).glob("*.hdf5")):
            # Only completed files are readable while a collector is active.
            if not path.name.startswith(split + "_"):
                continue
            try:
                with h5py.File(path) as dataset:
                    if "particle_nerd_manifest" not in dataset.attrs:
                        continue
                    manifest = json.loads(dataset.attrs["particle_nerd_manifest"])
                    if not manifest["complete"] or manifest["split"] != split:
                        continue
                    for name in dataset["data"]:
                        self.entries.append((path, name))
            except BlockingIOError:
                continue
        if not self.entries:
            raise ValueError(f"No completed {split} episodes in {directory}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        path, name = self.entries[index]
        with h5py.File(path) as dataset:
            manifest = json.loads(dataset.attrs["particle_nerd_manifest"])
            group = dataset["data"][name]
            episode = {key: value[:].astype(np.float32) for key, value in group["states"].items()}
            episode.update({key: value[0] for key, value in group["constants"].items()})
            episode.update(
                material=manifest["material"],
                gravity=np.asarray(manifest["gravity"]),
                body_shapes=manifest["body_shapes"],
                moving_body=manifest["moving_object_id"],
                name=f"{path.stem}/{name}",
                dt=float(episode["dt"]),
            )
            episode["force_reference"] = group["reference/body_impulse"][:].astype(float) / episode["dt"]
            episode["rigid_contact_impulse"] = group["reference/rigid_contact_impulse"][:]
        self.cache[index] = episode
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return episode


def velocity_history(episode, step, length=3):
    return np.concatenate([episode["particle_velocity"][max(0, step + i)] for i in range(1 - length, 1)], axis=1)
