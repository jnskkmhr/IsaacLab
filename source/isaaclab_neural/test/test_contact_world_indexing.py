# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contact-token and root-body Newton world indexing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.solvers.neural_solver import _resolve_root_body_ids


def test_resolve_root_body_ids_uses_articulation_joint_children():
    model = SimpleNamespace(
        body_count=6,
        joint_child=torch.tensor([5, 4, 2, 1]),
        body_world_start=torch.tensor([0, 3]),
    )

    root_body_ids = _resolve_root_body_ids(model, np.asarray([0, 2, 4]), num_envs=2)

    np.testing.assert_array_equal(root_body_ids, np.asarray([5, 2]))


def test_contact_token_world_ids_are_derived_from_owner_bodies():
    adapter = DataGenerationAdapter.__new__(DataGenerationAdapter)
    adapter.device = torch.device("cpu")
    adapter.model = SimpleNamespace(body_world=torch.tensor([1, 0, 1, 0]))
    adapter.contact_adapter = SimpleNamespace(contact_token_body_ids=torch.tensor([[1, -1], [0, 2]], dtype=torch.long))

    world_ids = adapter.contact_token_world_ids

    assert world_ids is not None
    torch.testing.assert_close(world_ids, torch.tensor([[0, -1], [1, 1]], dtype=torch.long))


@pytest.mark.parametrize("adapter_kind", ["dataset", "neural"])
def test_generalized_state_reset_refreshes_root_transform(monkeypatch, adapter_kind):
    """Direct state resets must refresh body poses even after FK masks were consumed."""
    import newton
    import warp as wp
    from isaaclab_neural.envs.neural_env_wrapper import NeuralEnvAdapter
    from isaaclab_neural.generate.adapter import NewtonDataGenerationBackend
    from isaaclab_neural.solvers.neural_solver import NeuralSolver
    from isaaclab_newton.physics import NewtonManager

    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_body(mass=1.0)
    builder.end_world()
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    monkeypatch.setattr(NewtonManager, "_model", model)
    monkeypatch.setattr(NewtonManager, "_state_0", state)
    monkeypatch.setattr(NewtonManager, "_world_reset_mask", wp.zeros(2, dtype=wp.bool, device="cpu"))
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", wp.zeros(1, dtype=wp.bool, device="cpu"))
    monkeypatch.setattr(NewtonManager, "_eval_fk", NewtonManager._eval_fk_impl)
    monkeypatch.setattr(NewtonManager, "_reset_solver_internals_delegate", lambda mask: None)
    monkeypatch.setattr(NewtonManager, "_mark_sensor_state_dirty", lambda: None)
    monkeypatch.setattr(NewtonManager, "_mark_transforms_dirty", lambda: None)

    solver = NeuralSolver.__new__(NeuralSolver)
    solver.num_envs = 1
    solver.dof_q_per_env = model.joint_coord_count
    solver.dof_qd_per_env = model.joint_dof_count
    newton.solvers.SolverBase.__init__(solver, model)
    solver.torch_device = torch.device("cpu")
    monkeypatch.setattr(NewtonManager, "_solver", solver)
    target = torch.cat((wp.to_torch(state.joint_q), wp.to_torch(state.joint_qd))).reshape(1, -1).clone()
    target[0, :3] = torch.tensor([1.0, 2.0, 3.0])

    if adapter_kind == "dataset":
        adapter = NewtonDataGenerationBackend.__new__(NewtonDataGenerationBackend)
        adapter._manager = NewtonManager
        adapter.state = state
        adapter.assign_solver_states(solver, target)
    else:
        adapter = NeuralEnvAdapter.__new__(NeuralEnvAdapter)
        adapter.manager = NewtonManager
        monkeypatch.setattr(adapter, "sync", lambda **kwargs: None)
        monkeypatch.setattr(adapter, "reset_history", lambda: None)
        adapter.reset(initial_states=target)

    torch.testing.assert_close(wp.to_torch(state.body_q)[0, :3], target[0, :3])
