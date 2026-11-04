# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Neural dynamics with fixed particle storage, recycling, resets and graph reuse.

Use the task's bundled checkpoint. Native MPM construction and stepping are
blocked throughout the real two-world simulation.
"""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.solvers import SolverImplicitMPM

from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsManager

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.visualizer import MovingPatchParticleRendering
from isaaclab_tasks.contrib.g1_29dof_particle_nerd_moving_patch.env_cfg.terrain_cfg import FLAT_TERRAINS_CFG
from isaaclab_tasks.contrib.particle_nerd.manager import NewtonParticleNeRDCouplerManager
from isaaclab_tasks.utils import resolve_task_config


def test_neural_recycling_reset_and_graph_reuse(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("NeRD integration requires CUDA")

    def forbidden(*args, **kwargs):
        raise AssertionError("Particle NeRD must neither construct nor step native MPM")

    monkeypatch.setattr(SolverImplicitMPM, "__init__", forbidden)
    monkeypatch.setattr(SolverImplicitMPM, "step", forbidden)
    task = "IsaacContrib-Velocity-Sand-G1-29dof-ParticleNeRD-MovingPatch"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=["env.scene.num_envs=2"])
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG.replace(num_rows=2, num_cols=2)
    for group in (cfg.curriculum, cfg.terminations):
        for name in vars(group):
            if not name.startswith("_"):
                setattr(group, name, None)
    cfg.events.push_robot = None
    cfg.events.reset_base.params = {"pose_range": {}, "velocity_range": {}}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    solver_cfg = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    solver_cfg.compile_model = False
    solver_cfg.active_k = 512
    cfg.sim.physics.capture_solver = False
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            solver = NewtonManager._solver.solver("sand")
            patch = env.moving_patch_particle
            action = torch.zeros((2, env.action_manager.total_action_dim), device=env.device)
            maximum_force = torch.zeros(2, device=env.device)
            for _ in range(12):
                _, reward, _, _, _ = env.step(action)
                assert torch.isfinite(reward).all()
                assert torch.isfinite(solver.history).all()
                maximum_force = torch.maximum(maximum_force, env.foot_contact_force.norm(dim=-1).amax(-1))
            assert (maximum_force > 1).all(), "Both worlds must receive actual contact reactions"

            state = NewtonManager.get_state_0()
            original_bodies = state.body_q.numpy().copy()
            world_zero = patch.model.particle_world.numpy() == 0
            body_zero = patch.model.body_world.numpy() == 0
            before_q = state.particle_q.numpy().copy()
            before_history = solver.history_wp.numpy().copy()
            before_reference = patch.particle_reset_positions.numpy().copy()
            center = patch.patch_centers.numpy()[0]
            central = world_zero & (np.abs(before_q[:, :2] - center) < 0.15).all(axis=1)
            central &= patch.particle_is_dynamic.numpy() != 0
            assert central.any()
            assert np.max(np.abs(before_history[central, -1] - before_history[central, 0])) > 1e-6

            moved = original_bodies.copy()
            moved[body_zero, 0] += 0.2
            state.body_q.assign(moved)
            patch.update(state)
            patch.check_terrain_queries()
            # Overlapping interior particles retain their actual deformations and neural history.
            np.testing.assert_array_equal(state.particle_q.numpy()[central], before_q[central])
            np.testing.assert_array_equal(solver.history_wp.numpy()[central], before_history[central])
            np.testing.assert_array_equal(state.particle_q.numpy()[~world_zero], before_q[~world_zero])
            np.testing.assert_array_equal(solver.history_wp.numpy()[~world_zero], before_history[~world_zero])
            entered = world_zero & (
                np.abs(patch.particle_reset_positions.numpy() - before_reference).max(axis=1) > 1e-5
            )
            assert entered.any()
            entered_q = state.particle_q.numpy()[entered]
            np.testing.assert_array_equal(state.particle_qd.numpy()[entered], 0)
            np.testing.assert_array_equal(solver.history_wp.numpy()[entered], np.repeat(entered_q[:, None], 3, axis=1))

            # Returning to an abandoned region creates fresh material; no old footprint is restored.
            marker = int(np.flatnonzero(central)[0])
            wp.to_torch(state.particle_q)[marker, 2] -= 0.03
            deformed = state.particle_q.numpy()[marker].copy()
            moved = original_bodies.copy()
            moved[body_zero, 0] += 3.2
            state.body_q.assign(moved)
            patch.update(state)
            state.body_q.assign(original_bodies)
            patch.update(state)
            returned = state.particle_q.numpy()[marker]
            np.testing.assert_allclose(returned, before_reference[marker], atol=1e-6, rtol=0)
            assert abs(returned[2] - deformed[2]) > 0.01
            np.testing.assert_array_equal(solver.history_wp.numpy()[marker], np.repeat(returned[None], 3, axis=0))

            class PointRecorder(MovingPatchParticleRendering):
                """Exercise the same particle submission path as GL and RTX."""

                def __init__(self):
                    self.show_particles = True
                    self.moving_patch_visualizer = SimpleNamespace(
                        moving_patch_particle=patch, get_visualized_env_ids=lambda: [0]
                    )
                    self.points = {}

                def log_points(self, name, points, **kwargs):
                    self.points[name] = points.numpy().copy()

            viewer = PointRecorder()
            viewer._log_particles(state)
            assert set(viewer.points) == {"/model/particles"}
            assert 0 < len(viewer.points["/model/particles"]) <= solver.capacity
            assert np.isfinite(viewer.points["/model/particles"]).all()

            other_history = solver.history[solver.capacity :].clone()
            env._reset_idx(torch.tensor([0], device=env.device))
            torch.testing.assert_close(solver.history[solver.capacity :], other_history, rtol=0, atol=0)
            patch.update(NewtonManager.get_state_0())
            PhysicsManager._cfg.capture_solver = True
            for _ in range(3):
                _, reward, _, _, _ = env.step(action)
                assert torch.isfinite(reward).all()
            graphs = dict(NewtonParticleNeRDCouplerManager._solver_graphs)
            assert graphs
            count = patch.model.particle_count
            history_buffer = solver.history.data_ptr()
            state = NewtonManager.get_state_0()
            original_bodies = state.body_q.numpy().copy()
            for offset in (0.4, 1.2, 2.0, 3.2, 0.0):
                moved = original_bodies.copy()
                moved[body_zero, 0] += offset
                state.body_q.assign(moved)
                patch.update(state)
                patch.check_terrain_queries()
                assert patch.model.particle_count == count
                assert solver.history.data_ptr() == history_buffer
                _, reward, _, _, _ = env.step(action)
                assert torch.isfinite(reward).all()
                assert torch.isfinite(solver.history).all()
                assert NewtonParticleNeRDCouplerManager._solver_graphs == graphs, "Travel must reuse captured graphs"
        finally:
            env.close()
