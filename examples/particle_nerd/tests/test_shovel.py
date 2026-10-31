# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""The concave shovel must remain hollow and expose differentiable collider features."""

import json

import numpy as np
import torch
import warp as wp

from particle_nerd.shovel.asset import ASSET_DIR
from particle_nerd.transolver.features import features
from particle_nerd.transolver.inference import manifest_colliders, manifest_sdf


def test_shovel_bowl_is_empty_and_geometry_gradient_is_finite():
    wp.init()
    m = json.loads((ASSET_DIR / "metadata.json").read_text())
    com = np.asarray(m["original_local_com"])
    # Center of the blade interior, 2 cm above its local upper surface; not a convex hull.
    points = np.array([[0.031, 0.011, 0.015], [0.031, 0.011, -0.012]]) - com
    history = torch.tensor(
        np.repeat(points[:, None], 3, axis=1), dtype=torch.float32, device="cuda:0", requires_grad=True
    )
    poses = torch.tensor([[0, 0, 0, 0, 0, 0, 1], [0, 0, -2, 0, 0, 0, 1]], dtype=torch.float32, device="cuda:0")
    velocity = torch.zeros(2, 6, device="cuda:0")
    colliders = manifest_colliders(m, "cuda:0")
    material = torch.zeros(13, device="cuda:0")

    def objective(h):
        _, pairs, weights = features(h, poses, velocity, *colliders, material, sdf=manifest_sdf(m, "cuda:0"))
        return pairs[0, 0, 0], pairs

    distance, pairs = objective(history)
    assert float(pairs[0, 0, 0].detach()) > 0
    assert float(pairs[1, 0, 0].detach()) < 0
    distance.backward()
    eps = 1e-5
    plus, minus = history.detach().clone(), history.detach().clone()
    plus[0, -1, 2] += eps
    minus[0, -1, 2] -= eps
    fd = (objective(plus)[0] - objective(minus)[0]) / (2 * eps)
    torch.testing.assert_close(history.grad[0, -1, 2], fd, atol=0.03, rtol=0.01)


def test_shovel_ground_contacts_are_not_truncated():
    """The multi-piece shovel can produce more than 128 contacts with the ground."""
    from particle_nerd.runner.rigid import NewtonRigidStepper

    meta = json.loads((ASSET_DIR / "metadata.json").read_text())
    # Ground-contact pose reported by the interactive demo at step 550.
    pose = [-0.0026004759, -0.10200842, 0.0807573, -0.009833644, -0.04169984, -0.46443614, 0.8845697]
    episode = dict(
        body_pose=np.array([[pose]], dtype=np.float32),
        body_velocity=np.zeros((1, 1, 6), dtype=np.float32),
        body_size=np.array([meta["body_size_m"]]),
        body_mass=np.array([meta["body_mass_kg"]]),
        body_inertia=np.array([meta["body_inertia"]]),
        body_com=np.zeros((1, 3)),
        body_shapes=["shovel"],
        body_friction=[0.5],
        gravity=[0, 0, -9.81],
        dt=0.005,
        ground_plane=True,
    )
    rigid = NewtonRigidStepper(episode)
    rigid.pipeline.collide(rigid.state, rigid.contacts)
    count = int(rigid.contacts.rigid_contact_count.numpy()[0])
    assert count > 128, "Regression scene must exceed the old rigid-contact capacity"
    pose, velocity = rigid.step(np.zeros((1, 6)))
    assert int(rigid.solver.mjw_data.nacon.numpy()[0]) == count, "Rigid contacts were truncated"
    assert not rigid.solver.mjw_data.overflow.numpy().any(), "MJWarp constraint storage overflowed"
    assert np.isfinite(pose).all() and np.isfinite(velocity).all()
