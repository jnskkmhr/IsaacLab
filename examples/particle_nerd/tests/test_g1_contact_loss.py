# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect contact-local supervision, world/local geometry, and the original default loss."""

import math

import pytest
import torch

from particle_nerd.g1.train import errors


def test_contact_loss_focuses_reference_contacts_with_transformed_foot():
    # A foot rotated 90 degrees around Z and translated; its collider is offset
    # by 3.5 cm along local X and -3.04 cm along local Z from the body frame.
    targets = torch.tensor([[[1.0, 2.035, 2.9546], [1.0, 3.035, 2.9696]]])
    sample = dict(
        targets=targets,
        history=targets[0, :, None].repeat(1, 3, 1),
        poses=torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]]]),
        velocities=torch.zeros(1, 1, 6),
        geometry=dict(
            body_ids=torch.tensor([0]),
            local_pose=torch.tensor([[0.035, 0.0, -0.0304, 0.0, 0.0, 0.0, 1.0]]),
            com=torch.zeros(1, 3),
            sizes=torch.tensor([[0.085, 0.03, 0.005]]),
            foot_slots=torch.tensor([0]),
        ),
    )
    prediction = targets.clone()
    prediction[0, 0, 0] += 0.01  # Near-contact particle: 1 cm error.
    prediction[0, 1, 0] += 0.1  # Distant particle: 10 cm error.
    prediction.requires_grad_(True)
    dt = 0.005

    # Both are stationary in the reference. Original mean error / dt² = 202.
    assert float(errors(prediction, sample, dt)[0].detach()) == pytest.approx(202.0, rel=1e-5)
    loss = errors(prediction, sample, dt, contact_weight=0.5, contact_band=0.02)[0]
    # Half the original loss plus half the near-contact error / dt² (4).
    assert float(loss.detach()) == pytest.approx(103.0, rel=1e-5)

    contact_only = errors(prediction, sample, dt, contact_weight=1.0, contact_band=0.02)[0]
    contact_only.backward()
    assert prediction.grad[0, 0, 0].item() == pytest.approx(800.0, rel=1e-5)
    assert torch.count_nonzero(prediction.grad[0, 1]).item() == 0
    assert float(errors(targets, sample, dt, contact_weight=0.5)[0]) == 0.0
