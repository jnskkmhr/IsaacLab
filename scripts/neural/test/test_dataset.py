# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the episode file and the windows that the network is trained on."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402

MATERIAL = {
    "density": 1600.0,
    "young_modulus": 1e15,
    "poisson_ratio": 0.3,
    "friction": 0.48,
    "yield_pressure": 1e15,
    "tensile_yield_ratio": 0.0,
    "yield_stress": 0.0,
    "hardening": 0.0,
    "dilatancy": 0.0,
    "viscosity": 0.0,
}


def write_episode(root: Path, category: str, index: int, frames: int, with_body: bool) -> dict:
    """Write an episode whose arrays hold the frame number, so that a window shows which frames it took."""
    record = dataset.episode("flat", category, index)
    record["duration_s"] = (frames - 1) * dataset.DT
    time = np.arange(frames, dtype=np.float32)
    positions = np.zeros((frames, 4, 3), dtype=np.float32)
    positions[:, :, 2] = time[:, None]
    body = None
    attrs = {"particle_spacing_m": 0.032, "particle_mass_kg": 0.05, "material": MATERIAL, "bed_size_m": [2.0, 1.2, 0.3]}
    if with_body:
        pose = np.tile(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), (frames, 1))
        pose[:, 0] = time
        vel = np.zeros((frames, 6), dtype=np.float32)
        vel[:, 0] = time
        wrench = np.zeros((frames - 1, 6), dtype=np.float32)
        body = {"pose": pose, "vel": vel, "ext_wrench": wrench, "contact_wrench": wrench}
        attrs["object_size_m"] = record["object_size_m"]
    path = root / record["split"] / "flat" / category / f"{index:04d}.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_episode(path, record, positions, body, attrs)
    return record


@pytest.mark.parametrize("category, first_body_frame", [("lateral_push", 6), ("impact", 5)])
def test_window_gives_the_collider_state_that_the_mpm_step_saw(tmp_path, category, first_body_frame):
    """A kinematic object is given by its state after the step, a free object by its state before the step."""
    write_episode(tmp_path, category, 0, frames=12, with_body=True)
    windows = dataset.EpisodeWindows(tmp_path, "train")

    sample = windows.sequence(np.random.default_rng(0), 4, "cpu", index=0, start=5)

    assert sample["history"][0, :, 2].tolist() == [3.0, 4.0, 5.0]
    assert sample["targets"][:, 0, 2].tolist() == [6.0, 7.0, 8.0, 9.0]
    expected = [float(first_body_frame + k) for k in range(4)]
    assert sample["poses"][:, 0, 0].tolist() == expected
    assert sample["velocities"][:, 0, 0].tolist() == expected


def test_free_object_gets_its_release_velocity_at_the_release_step(tmp_path):
    """The recorded velocity at the release step is still zero; the window restores the velocity of the throw."""
    record = write_episode(tmp_path, "impact", 1, frames=dataset.HOVER_STEPS + 4, with_body=True)
    windows = dataset.EpisodeWindows(tmp_path, "train")

    sample = windows.sequence(np.random.default_rng(0), 3, "cpu", index=0, start=dataset.HOVER_STEPS - 1)

    released = torch.tensor(record["release_velocity"], dtype=torch.float32)
    assert torch.equal(sample["velocities"][1, 0], released)
    assert sample["velocities"][0, 0, 0] == dataset.HOVER_STEPS - 1


def test_sand_only_window_parks_a_box_far_away_and_repeats_the_first_frame(tmp_path):
    write_episode(tmp_path, "rest_bed", 0, frames=8, with_body=False)
    windows = dataset.EpisodeWindows(tmp_path, "train")

    sample = windows.sequence(np.random.default_rng(0), 3, "cpu", index=0, start=0)

    assert sample["poses"][:, 0, :3].tolist() == [[0.0, 0.0, 10.0]] * 3
    assert sample["history"][0, :, 2].tolist() == [0.0, 0.0, 0.0]
    assert sample["targets"][:, 0, 2].tolist() == [1.0, 2.0, 3.0]


def test_unfinished_files_are_left_out(tmp_path):
    """A file that was not finished must not be trained on."""
    write_episode(tmp_path, "rest_bed", 0, frames=8, with_body=False)
    record = write_episode(tmp_path, "rest_bed", 1, frames=8, with_body=False)
    path = tmp_path / record["split"] / "flat" / "rest_bed" / "0001.hdf5"
    with h5py.File(path, "a") as handle:
        handle.attrs["complete"] = False

    windows = dataset.EpisodeWindows(tmp_path, "train")

    assert [r[0]["id"] for r in windows.records] == ["flat-rest_bed-0000"]
