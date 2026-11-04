# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed particle/rigid worlds share policy observations and reset independently."""

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.solvers import SolverImplicitMPM

from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsManager

from isaaclab_tasks.contrib.g1_29dof_particle_nerd_moving_patch.env_cfg.terrain_cfg import FLAT_TERRAINS_CFG
from isaaclab_tasks.contrib.particle_nerd.manager import NewtonParticleNeRDCouplerManager
from isaaclab_tasks.utils import resolve_task_config


def test_mixed_contact_recycling_and_independent_resets(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Particle NeRD integration requires CUDA")

    def forbidden(*args, **kwargs):
        raise AssertionError("Mixed Particle NeRD must not construct or step native MPM")

    monkeypatch.setattr(SolverImplicitMPM, "__init__", forbidden)
    monkeypatch.setattr(SolverImplicitMPM, "step", forbidden)
    task = "IsaacContrib-Velocity-G1-29dof-ParticleNeRD-MovingPatch-MixedTerrain-Play"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG.replace(num_cols=2)
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.sim.physics.capture_solver = False
    neural = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    neural.compile_model = False
    neural.active_k = 512
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            assert all(torch.isfinite(value).all() and value.shape[0] == 4 for value in obs.values())
            model = NewtonManager.get_model()
            solver = NewtonManager._solver.solver("sand")
            patch = env.moving_patch_particle
            np.testing.assert_array_equal(np.unique(model.particle_world.numpy()), [0, 1])
            assert env.scene["sand"].num_instances == 2
            np.testing.assert_array_equal(env.scene["sand"].scene_env_ids.cpu().numpy(), [0, 1])
            np.testing.assert_array_equal(env.scene.terrain.mpm_env_mask.cpu().numpy(), [True, True, False, False])
            assert len(solver.slots) == 2
            assert len(solver.estimator.feet) == 4
            assert solver.history.shape[0] == model.particle_count
            bodies = model.shape_body.numpy()
            margins = model.shape_margin.numpy()
            for world, margin in enumerate((0.02, 0.02, 0.0, 0.0)):
                feet = env._foot_body_ids[world].cpu().numpy()
                np.testing.assert_allclose(margins[np.isin(bodies, feet)], margin, atol=1e-7)

            action = torch.zeros((4, env.action_manager.total_action_dim), device=env.device)
            maximum_force = torch.zeros(4, device=env.device)
            network_calls = []
            contact_calls = []
            original_forward = solver.network.forward
            original_launch = wp.launch

            def count_contacts(*args, **kwargs):
                kernel = kwargs.get("kernel", args[0] if args else None)
                if kernel.key in ("solve_contacts_parallel", "solve_contacts_batched"):
                    contact_calls.append(kernel.key)
                return original_launch(*args, **kwargs)

            def compare_batch(*inputs):
                network_calls.append(inputs[0].shape)
                output = original_forward(*inputs)
                if inputs[0].ndim == 3:
                    reference = torch.stack([original_forward(*(value[i] for value in inputs)) for i in range(2)])
                    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-5)
                return output

            with monkeypatch.context() as probe:
                probe.setattr(solver.network, "forward", compare_batch)
                probe.setattr(wp, "launch", count_contacts)
                env.step(action)
            assert len(network_calls) == cfg.decimation, "Compatible sand worlds must share one neural call per step"
            assert all(shape[0] == 2 and len(shape) == 3 for shape in network_calls)
            assert len(contact_calls) == cfg.decimation, "Independent feet must share one contact solve per step"

            for _ in range(30):
                obs, reward, _, _, _ = env.step(action)
                assert torch.isfinite(reward).all() and torch.isfinite(solver.history).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                maximum_force = torch.maximum(maximum_force, env.foot_contact_force.norm(dim=-1).amax(-1))
            assert (maximum_force > 1.0).all(), "Sand and rigid worlds must both report supporting contact forces"
            estimator = solver.estimator
            original_evaluate = estimator.evaluate
            contact_counts = []

            def compare_contacts(*args, **kwargs):
                result = original_evaluate(*args, **kwargs)
                batched = wp.to_torch(result).clone()
                counts = torch.stack([wp.to_torch(foot["stats"])[0].clone() for foot in estimator.feet])
                estimator.batch_contacts = False
                estimator.captured = True
                try:
                    reference = original_evaluate(*args, **kwargs)
                    torch.testing.assert_close(wp.to_torch(reference), batched, rtol=1e-6, atol=1e-5)
                    sequential_counts = torch.stack([wp.to_torch(foot["stats"])[0] for foot in estimator.feet])
                    torch.testing.assert_close(sequential_counts, counts, rtol=0, atol=0)
                    contact_counts.append(counts)
                finally:
                    estimator.batch_contacts = True
                    wp.to_torch(result).copy_(batched)
                return result

            with monkeypatch.context() as probe:
                probe.setattr(estimator, "evaluate", compare_contacts)
                env.step(action)
            assert torch.stack(contact_counts).sum() > 0, "Force equivalence must include material contacts"
            feedback = wp.to_torch(env._coupling_forces)[env._foot_body_ids[2:]]
            torch.testing.assert_close(feedback, torch.zeros_like(feedback), rtol=0, atol=0)

            history = solver.history.clone()
            particles = NewtonManager.get_state_0().particle_q.numpy().copy()
            env._reset_idx(torch.tensor([2], device=env.device))
            patch.update(NewtonManager.get_state_0())
            torch.testing.assert_close(solver.history, history, rtol=0, atol=0)
            np.testing.assert_array_equal(NewtonManager.get_state_0().particle_q.numpy(), particles)
            assert not env.foot_contact_force[2].any(), "Rigid reset must invalidate stale contact reactions"
            env._reset_idx(torch.tensor([0], device=env.device))
            patch.update(NewtonManager.get_state_0())
            expected = wp.to_torch(NewtonManager.get_state_0().particle_q)[solver.slots[0]]
            torch.testing.assert_close(
                solver.history[solver.slots[0]], expected[:, None].expand(-1, 3, -1), rtol=0, atol=0
            )
            torch.testing.assert_close(solver.history[solver.slots[1]], history[solver.slots[1]], rtol=0, atol=0)
            np.testing.assert_array_equal(env.scene.terrain.mpm_env_mask.cpu().numpy(), [True, True, False, False])

            PhysicsManager._cfg.capture_solver = True
            for _ in range(3):
                env.step(action)
            graphs = dict(NewtonParticleNeRDCouplerManager._solver_graphs)
            assert graphs
            count = model.particle_count
            history_buffer = solver.history.data_ptr()
            before = patch.particle_reset_positions.numpy().copy()
            state = NewtonManager.get_state_0()
            shifted = state.body_q.numpy().copy()
            shifted[model.body_world.numpy() == 0, 0] += 0.4
            state.body_q.assign(shifted)
            patch.update(state)
            patch.check_terrain_queries()
            assert (patch.particle_reset_positions.numpy() != before).any(), "Moving sand must recycle particle slots"
            _, reward, _, _, _ = env.step(action)
            assert torch.isfinite(reward).all()
            assert model.particle_count == count and solver.history.data_ptr() == history_buffer
            assert NewtonParticleNeRDCouplerManager._solver_graphs == graphs
        finally:
            env.close()
