# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mechanics contracts missing from the force-balance-only tests.

Protect constitutive preload, moving-surface friction/torque, and contact memory.
These use the evaluator's public assembly/fit API and independently known loads.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dem_reconstruction import ContactHistory, DemParameters, assemble_dem_frame, fit_dem_frame


def frame(position, velocity, poses, body_velocity=None, *, history=None, parameters=None):
    poses = np.asarray(poses, dtype=float)
    return assemble_dem_frame(
        np.asarray([position]),
        np.asarray([velocity]),
        np.ones(1),
        np.full(1, 0.016),
        poses,
        np.zeros((len(poses), 6)) if body_velocity is None else np.asarray(body_velocity),
        np.tile([1, 1, 0.2], (len(poses), 1)),
        np.ones(len(poses)),
        np.zeros((1, 3)),
        dt=0.1,
        parameters=parameters or DemParameters(normal_stiffness=100, body_offset=0.01),
        history=history if history is not None else ContactHistory(),
        gravity=(0, 0, 0),
    )


def test_contact_law_recovers_preload_invisible_to_particle_acceleration():
    # Two opposed springs compressed 1 cm, stiffness 100 N/m: 1 N on each body,
    # even though particle acceleration and net contact force are both zero.
    f = frame([0, 0, 0], [0, 0, 0], [[0, 0, -0.1, 0, 0, 0, 1], [0, 0, 0.1, 0, 0, 0, 1]])
    result = fit_dem_frame(f)
    assert result["converged"]
    np.testing.assert_allclose(result["body_wrench"][:, 2], [-1, 1], atol=1e-5)
    np.testing.assert_allclose(f.direct_body_wrench[:, 2], [-1, 1], atol=1e-12)
    zero = fit_dem_frame(f, zero_prior=True)
    np.testing.assert_allclose(zero["body_wrench"], 0, atol=1e-12)


def test_rotating_body_surface_sets_friction_and_moment_about_body_center():
    # omega_z=1 at x=.2 gives surface velocity +.2 in y. Damping=4 Ns/m:
    # particle friction +.8 N in y; body reaction -.8 N in y and -1 N in z.
    p = DemParameters(normal_stiffness=100, body_offset=0.01, friction_model="velocity")
    f = frame([0.2, 0, 0], [0, 0, 0], [[0, 0, -0.1, 0, 0, 0, 1]], [[0, 0, 0, 0, 0, 1]], parameters=p)
    np.testing.assert_allclose(f.direct_body_wrench[0], [0, -0.8, -1, 0.08, 0.2, -0.16], atol=1e-12)
    translated = frame([0.2, 0, 0], [3, 0, 0], [[0, 0, -0.1, 0, 0, 0, 1]], [[3, 0, 0, 0, 0, 1]], parameters=p)
    np.testing.assert_allclose(translated.direct_body_wrench, f.direct_body_wrench, atol=1e-12)


def test_tangential_history_supports_static_friction_and_resets_after_separation():
    h = ContactHistory()
    p = DemParameters(normal_stiffness=100, damping_ratio=0, body_offset=0.01)
    pose = [[0, 0, -0.1, 0, 0, 0, 1]]
    sliding = frame([0, 0, 0], [0.1, 0, 0], pose, history=h, parameters=p)
    stopped = frame([0, 0, 0], [0, 0, 0], pose, history=h, parameters=p)
    np.testing.assert_allclose(sliding.direct_body_wrench[0, :3], [0.5, 0, -1], atol=1e-12)
    np.testing.assert_allclose(stopped.direct_body_wrench[0, :3], [0.5, 0, -1], atol=1e-12)
    separated = frame([0, 0, 0.1], [0, 0, 0], pose, history=h, parameters=p)
    np.testing.assert_allclose(separated.direct_body_wrench, 0, atol=1e-12)
    recontact = frame([0, 0, 0], [0, 0, 0], pose, history=h, parameters=p)
    np.testing.assert_allclose(recontact.direct_body_wrench[0, :3], [0, 0, -1], atol=1e-12)


def test_pair_contact_transmits_equal_and_opposite_internal_forces():
    # Two unit masses separated by .022 m, effective rest distance .032 m.
    # A 100 N/m spring supplies 1 N outward to each, giving opposite 1 m/s^2 accelerations.
    f = assemble_dem_frame(
        np.array([[-0.011, 0, 0], [0.011, 0, 0]]),
        np.zeros((2, 3)),
        np.ones(2),
        np.full(2, 0.016),
        np.empty((0, 7)),
        np.empty((0, 6)),
        np.empty((0, 3)),
        np.empty(0),
        np.array([[-1, 0, 0], [1, 0, 0]]),
        dt=0.01,
        parameters=DemParameters(normal_stiffness=100, damping_ratio=0),
        history=ContactHistory(),
        gravity=(0, 0, 0),
    )
    result = fit_dem_frame(f)
    assert result["converged"]
    assert result["relative_residual"] < 1e-5
    assert result["law_rmse_N"] < 1e-5
