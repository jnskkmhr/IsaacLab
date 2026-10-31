# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a Transolver checkpoint and settled particle history; no training dependency."""

import json

import h5py
import numpy as np
import torch


def material_features(manifest, mass, radius, dt):
    """Checkpoint-compatible material normalization, including particle mass/radius."""
    m = manifest["material"]
    return np.asarray(
        [
            m["density"] / 1600,
            np.log10(m["young_modulus"]) / 15,
            m["poisson_ratio"],
            m["friction"],
            np.log1p(m["yield_pressure"]) / np.log1p(1e15),
            m["tensile_yield_ratio"],
            np.log1p(m["yield_stress"]) / 15,
            m["hardening"],
            m["dilatancy"],
            np.log1p(m["viscosity"]) / 10,
            mass / 0.05,
            radius / 0.02,
            dt / 0.02,
        ],
        dtype=np.float32,
    )


def load(checkpoint, episode, environments=1):
    """Replicate one settled episode; callers may change individual initial states."""
    if environments < 1:
        raise ValueError("environments must be positive")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("architecture") not in ("particle_transolver_v1", "particle_contact_transolver_v1"):
        raise ValueError("Only particle_transolver_v1 or particle_contact_transolver_v1 checkpoints are supported")
    with h5py.File(episode) as f:
        manifest = json.loads(f.attrs["manifest"])
        frames = f["frame_steps"][:]
        start = int(np.searchsorted(frames, manifest["settled_step"]))
        h = saved["config"]["model"]["history"]
        if start < h - 1 or start >= len(frames):
            raise ValueError("Episode lacks settled particle history")
        dt = saved["config"]["dt"]
        if not np.isclose(dt, 1 / manifest["record_hz"]):
            raise ValueError("Checkpoint and episode particle timesteps differ")
        if saved["config"].get("geometry", "box") != manifest.get("geometry", "box"):
            raise ValueError("Checkpoint and episode geometries differ")
        history = np.stack([f["particle_position"][j] for j in range(start - h + 1, start + 1)], axis=1)
        mass, radius = f["particle_mass"][:], f["particle_radius"][:]
        if not np.allclose(mass, mass[0]) or not np.allclose(radius, radius[0]):
            raise ValueError("Checkpoint material features require uniform particle mass and radius per world")
        step = frames[start]
        initial = dict(
            history=history,
            particle_mass=mass,
            material=material_features(manifest, float(mass[0]), float(radius[0]), dt),
            body_pose=f["body_pose"][step],
            body_velocity=f["body_velocity"][step],
        )
    return saved, manifest, {key: np.repeat(value[None], environments, axis=0) for key, value in initial.items()}
