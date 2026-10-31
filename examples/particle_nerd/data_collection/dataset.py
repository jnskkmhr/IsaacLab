# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the current 50 Hz particle / 200 Hz rigid-body episode format."""

import json

import h5py
import numpy as np


def load_initial_state(path):
    """Read the settled initial state and measured particle velocity history."""
    with h5py.File(path, "r") as f:
        manifest = json.loads(f.attrs["manifest"])
        frames = f["frame_steps"][:]
        start = int(np.searchsorted(frames, manifest["settled_step"]))
        step = frames[start]
        return dict(
            manifest=manifest,
            frame=start,
            position=f["particle_position"][start],
            velocity=f["particle_velocity"][start],
            particle_mass=f["particle_mass"][:],
            particle_radius=f["particle_radius"][:],
            body_pose=f["body_pose"][step],
            body_velocity=f["body_velocity"][step],
        )


def initialize_predictor(checkpoint, episode, selector=None, compile_model=False, *, bed_size=None):
    """Construct the Transolver predictor without importing IsaacLab simulation modules."""
    import torch

    if selector is not None:
        raise ValueError("ParticleTransolver does not use a legacy GNN selector")
    state = load_initial_state(episode)
    if not np.allclose(state["particle_mass"], state["particle_mass"][0]) or not np.allclose(
        state["particle_radius"], state["particle_radius"][0]
    ):
        raise ValueError("Uniform particle mass and radius are required")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("architecture") != "particle_transolver_v1":
        raise ValueError(f"Expected particle_transolver_v1 checkpoint, got {saved.get('architecture')}")
    predictor = _initialize_transolver(saved, episode, state, compile_model, bed_size)
    if not np.isclose(predictor.dt, 1 / state["manifest"]["record_hz"]):
        raise ValueError("Checkpoint interval differs from the recorded particle interval")
    return predictor, state


def initialize_fresh_bed(state, bed_size):
    """Initialize the same fresh particle block and shovel pose for either physics backend."""
    manifest = state["manifest"]
    import warp as wp

    size = np.asarray(bed_size, dtype=np.float64)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Bed dimensions must be three finite positive values")
    scene = manifest["scene"]
    if scene["scene"] != "open-ground" or scene["ground_z_m"] != 0:
        raise ValueError("Fresh Transolver beds require an open scene with rigid ground at z=0")
    resolution = np.ceil(size / np.asarray(scene["particle_spacing_m"])).astype(int)
    spacing = size / resolution
    axes = [(np.arange(n) + 0.5) * step for n, step in zip(resolution, spacing, strict=True)]
    position = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    position[:, :2] -= size[:2] / 2
    position += np.random.default_rng(0).uniform(-0.2, 0.2, position.shape) * spacing
    position = position.astype(np.float32)
    n = len(position)
    state["position"] = position
    state["velocity"] = np.zeros_like(position)
    state["particle_mass"] = np.full(n, manifest["material"]["density"] * np.prod(spacing), dtype=np.float32)
    state["particle_radius"] = np.full(n, state["particle_radius"][0], dtype=np.float32)
    state["body_pose"] = state["body_pose"].copy()
    rotation = wp.quat_to_matrix(wp.quat(*state["body_pose"][3:]))
    half_height = np.abs(np.asarray(rotation).reshape(3, 3)[2]) @ (np.asarray(manifest["body_size_m"]) * 0.5)
    state["body_pose"][2] = size[2] + half_height + 2 * float(state["particle_radius"][0])
    state["body_velocity"] = np.zeros_like(state["body_velocity"])
    manifest["particle_count"] = n
    scene.update(
        bed_size_m=size.tolist(),
        lattice_resolution=resolution.tolist(),
        particle_spacing_m=spacing.tolist(),
        expected_particles_per_environment=n,
        initial_material_surface_z_m=float(size[2]),
        initialization="fresh jittered block at rest; no MPM settling",
    )


def _initialize_transolver(saved, episode, state, compile_model, bed_size):
    """Use settled history, or initialize an optional fresh open-ground particle block."""
    import torch

    from particle_nerd.transolver.features import material_features
    from particle_nerd.transolver.inference import Predictor, manifest_colliders

    manifest = state["manifest"]
    if saved["config"].get("geometry", "box") != manifest.get("geometry", "box"):
        raise ValueError("Checkpoint and initial episode use different rigid geometry")
    count = saved["config"]["model"]["history"]
    start = state["frame"]
    if bed_size is None and start < count - 1:
        raise ValueError("Episode does not contain enough pre-settling position history")
    if bed_size is None:
        with h5py.File(episode, "r") as f:
            history = np.stack([f["particle_position"][i] for i in range(start - count + 1, start + 1)], axis=1)
    else:
        initialize_fresh_bed(state, bed_size)
        position = state["position"]
        history = np.repeat(position[:, None, :], count, axis=1)
    predictor = Predictor(
        saved,
        torch.as_tensor(history, device="cuda:0"),
        torch.as_tensor(
            material_features(manifest, float(state["particle_mass"][0]), float(state["particle_radius"][0])),
            device="cuda:0",
        ),
        *manifest_colliders(manifest, "cuda:0"),
    )
    if compile_model:
        predictor.model = torch.compile(predictor.model)
    return predictor
