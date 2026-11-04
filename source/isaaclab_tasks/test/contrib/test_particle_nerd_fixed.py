# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-bed neural coupling, environment-coordinate invariance and independent resets."""

import hashlib
import os
from pathlib import Path

import gymnasium as gym
import newton
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sim.spawners.mpm.mpm import _generate_grid_particles
from newton.solvers import SolverImplicitMPM

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config


def test_recorded_demo_initial_particles_and_force_parameters():
    """Protect the archived zero-warm-up flat demo, not upstream task defaults."""
    cfg, _ = resolve_task_config("IsaacContrib-Velocity-Sand-Flat-G1-29dof-ParticleNeRD-Play", "rsl_rl_cfg_entry_point")
    cfg.validate()
    positions, velocities, masses, radii = _generate_grid_particles(cfg.scene.sand.spawn)
    # Independent reference: video/multi_terrain_5f1xyc21_zero_warmup_50fps/flat/metadata.json.
    assert hashlib.sha256(positions.tobytes()).hexdigest() == (
        "264d82ac061536a24c60fedce38036aae5675c7ad35d2a2f86542d815ec235d0"
    )
    assert not velocities.any()
    assert len(positions) == 890625
    assert float(masses[0]) == 0.006467368453741074
    assert float(radii[0]) == 0.007964757271111012
    assert cfg.scene.sand.spawn.material.density == 1600.0
    assert cfg.scene.sand.spawn.material.friction == 0.48
    solver = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    assert (solver.effective_mass_scale, solver.tangent_mass_scale) == (100.0, 40.0)
    assert (solver.horizontal_mass_scale, solver.upward_mass_scale, solver.normal_power) == (8.0, 2.0, 3.0)
    assert (solver.contact_iterations, solver.contact_relaxation) == (200, 0.02)
    assert (solver.active_k, solver.anchor_particles, solver.selection_interval) == ("auto", 512, 4)


@pytest.mark.parametrize("variant", ["Velocity-Sand-Flat", "Balance-Sand-Tilting"])
def test_fixed_bed_coordinates_contact_and_reset(monkeypatch, variant):
    checkpoint = os.environ.get("PARTICLE_NERD_CHECKPOINT", "")
    if not checkpoint or not Path(checkpoint).is_file() or not torch.cuda.is_available():
        pytest.skip("Set PARTICLE_NERD_CHECKPOINT and provide CUDA for the trained-solver integration test")

    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed Particle NeRD must not construct or step native MPM")

    monkeypatch.setattr(SolverImplicitMPM, "__init__", forbidden)
    monkeypatch.setattr(SolverImplicitMPM, "step", forbidden)
    task = f"IsaacContrib-{variant}-G1-29dof-ParticleNeRD-Play"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=["env.scene.num_envs=2"])
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.sim.physics.capture_solver = False
    solver_cfg = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    solver_cfg.checkpoint = checkpoint
    solver_cfg.compile_model = False
    solver_cfg.active_k = 1024
    # Start both robots directly above the particle bed to exercise neural feedback.
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.76)
    cfg.events.reset_base.params = {"pose_range": {}, "velocity_range": {}}
    if "Tilting" in variant:
        cfg.testbed_motion.debug_vis = False
        cfg.testbed_motion.settling_time = 0.0
        cfg.testbed_motion.ramp_time = 0.05
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            solver = NewtonManager._solver.solver("sand")
            entry = NewtonManager._solver._entries["sand"]
            # Newton assigns articulated virtual inertia after constructing the neural entry.
            # Contact reconstruction must use those masses, not the original foot-link masses.
            for foot in solver.estimator.feet:
                assert foot["mass"] == float(solver.model.body_mass.numpy()[foot["body"]])
            master = NewtonManager.get_model()
            assert int(newton.GeoType.CONVEX_MESH) in master.shape_type.numpy()

            first_collider_velocity = []
            if variant == "Velocity-Sand-Flat":
                from isaaclab_tasks.contrib.particle_nerd import solver as solver_module

                original_inference = solver_module.prepare_compact_step

                def observe_inference(history, poses, velocities, geometry, *args, **kwargs):
                    if not first_collider_velocity:
                        first_collider_velocity.append(velocities[geometry["foot_slots"]].clone())
                    return original_inference(history, poses, velocities, geometry, *args, **kwargs)

                monkeypatch.setattr(solver_module, "prepare_compact_step", observe_inference)
            bed = solver.fixed_bed
            state = entry.state_0
            history = solver.history[solver.slots[0]].clone()
            translated = history + bed.origins[1] - bed.origins[0]
            h0, p0, v0 = bed.inputs(history, state, 0, solver.geometry[0])
            h1, p1, v1 = bed.inputs(translated, state, 1, solver.geometry[1])
            torch.testing.assert_close(h0, h1, atol=2e-6, rtol=0)
            torch.testing.assert_close(p0, p1, atol=2e-6, rtol=0)
            torch.testing.assert_close(v0, v1, atol=2e-6, rtol=0)
            torch.testing.assert_close(bed.to_world(h0[:, -1], state, 0, 0.0), history[:, -1])

            action = torch.zeros((2, env.action_manager.total_action_dim), device=env.device)
            force = torch.zeros(2, device=env.device)
            for _ in range(20):
                obs, reward, _, _, _ = env.step(action)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(solver.history).all()
                force = torch.maximum(force, env.foot_contact_force.norm(dim=-1).amax(-1))
            if variant == "Velocity-Sand-Flat":
                # The reset robot is stationary at the first physics input. Its proxy already
                # has a nonzero free velocity after rigid integration; that is not the NN input.
                torch.testing.assert_close(first_collider_velocity[0], torch.zeros_like(first_collider_velocity[0]))
            assert (force > 1.0).all(), "Both robots must receive foot reactions while above the particle bed"
            if "Tilting" in variant:
                tray = wp.to_torch(entry.state_0.body_q)[bed.tray_ids]
                assert torch.linalg.vector_norm(tray[:, 3:6], dim=-1).min() > 1e-4, "The powered tray must tilt"
            untouched = solver.history[solver.slots[1]].clone()
            env.reset_sand_bed(torch.tensor([0], device=env.device))
            torch.testing.assert_close(solver.history[solver.slots[1]], untouched, rtol=0, atol=0)
            positions = wp.to_torch(NewtonManager.get_state_0().particle_q)[solver.slots[0]]
            torch.testing.assert_close(solver.history[solver.slots[0]], positions[:, None].expand(-1, 3, -1))
        finally:
            env.close()
