# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the evaluation rollout."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rollout  # noqa: E402

DT = 0.005
SPEED = torch.tensor([[0.005, 0.0, 0.0], [0.5, 0.0, 0.0]])
"""Speed [m/s] that the stand-in network gives its two particles: one below and one above 1 cm/s."""


class ConstantSpeed:
    """Stand-in for the network: every particle moves at its speed of :data:`SPEED`."""

    def step_from_history(self, history, pose, velocity, kinds, sizes, friction, material, dt, return_vel):
        return history[:, -1] + dt * SPEED, SPEED.clone()


def window(steps: int) -> dict:
    return {
        "history": torch.zeros((2, 3, 3)),
        "poses": torch.zeros((steps, 6, 7)),
        "velocities": torch.zeros((steps, 6, 6)),
        "kinds": torch.ones(6, dtype=torch.int32),
        "sizes": torch.ones((6, 3)),
        "friction": torch.ones(6),
        "material": torch.zeros(13),
        "dt": DT,
    }


def test_sleeping_holds_slow_particles_and_leaves_fast_ones_alone():
    awake, failure = rollout.predict(ConstantSpeed(), window(10))
    asleep, _ = rollout.predict(ConstantSpeed(), window(10), sleep_speed=0.01)

    assert failure is None
    np.testing.assert_allclose(awake[-1, :, 0], 10 * DT * SPEED[:, 0].numpy(), rtol=1e-5)
    assert (asleep[:, 0] == 0.0).all()  # exactly where it started, in every frame
    np.testing.assert_array_equal(asleep[:, 1], awake[:, 1])


def test_rollout_stops_when_a_particle_leaves_the_scene():
    class Escape(ConstantSpeed):
        def step_from_history(self, history, *args, **kwargs):
            return history[:, -1] + 3.0, SPEED.clone()

    positions, failure = rollout.predict(Escape(), window(10))

    assert failure is not None and failure["time_s"] == 2 * DT
    assert len(positions) == 2  # the start frame and the one step that stayed in bounds
