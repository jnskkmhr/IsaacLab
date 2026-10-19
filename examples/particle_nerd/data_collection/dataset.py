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


def initialize_predictor(checkpoint, episode, selector=None, compile_model=False):
    """Construct the current predictor without importing IsaacLab simulation modules."""
    import torch

    from particle_nerd.gnn.inference import ParticlePredictor
    from particle_nerd.gnn.region_selection import SelectedPredictor

    state = load_initial_state(episode)
    if not np.allclose(state["particle_mass"], state["particle_mass"][0]) or not np.allclose(
        state["particle_radius"], state["particle_radius"][0]
    ):
        raise ValueError("The current GNN expects uniform particle mass and radius")
    cls = SelectedPredictor if selector else ParticlePredictor
    options = dict(selector_checkpoint=selector, mode="learned") if selector else {}
    predictor = cls(
        checkpoint,
        state["position"],
        state["velocity"],
        state["manifest"],
        float(state["particle_mass"][0]),
        float(state["particle_radius"][0]),
        compile_model=compile_model,
        **options,
    )
    with h5py.File(episode, "r") as f:
        start = state["frame"]
        history = np.concatenate(
            [f["particle_velocity"][max(0, start - k)] for k in range(predictor.model.config["history"] - 1, -1, -1)],
            axis=-1,
        )
    predictor.history.copy_(torch.as_tensor(history, device=predictor.device))
    if not np.isclose(predictor.dt, 1 / state["manifest"]["record_hz"]):
        raise ValueError("Checkpoint interval differs from the recorded particle interval")
    return predictor, state
