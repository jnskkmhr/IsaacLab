# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-scene parity and independent, MPM-free object NeRD worlds."""

import gymnasium as gym
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.solvers import SolverImplicitMPM

from isaaclab.app import launch_simulation

from isaaclab_tasks.contrib.particle_nerd.network.model import ParticleTransolver
from isaaclab_tasks.contrib.particle_nerd.network.object_inference import PIPELINE, SCHEMA
from isaaclab_tasks.utils import resolve_task_config


@pytest.mark.parametrize("scene", ["Box", "SnowBall"])
def test_reference_scene_parity(scene):
    """Changing the material predictor must preserve the user's reference task settings."""
    reference, _ = resolve_task_config(f"IsaacContrib-ParticleWorld-{scene}-v0", "", overrides=[])
    nerd, _ = resolve_task_config(f"IsaacContrib-ParticleNeRD-{scene}-v0", "", overrides=[])
    reference = reference.to_dict()
    nerd = nerd.to_dict()
    physics = nerd["sim"]["physics"]
    original_physics = reference["sim"]["physics"]
    solver = physics["solver_cfg"]["entries"][1]["solver_cfg"]
    original_solver = original_physics["solver_cfg"]["entries"][1]["solver_cfg"]
    g1, _ = resolve_task_config("IsaacContrib-Velocity-Sand-Flat-G1-29dof-ParticleNeRD-Play", "", overrides=[])
    g1_solver = g1.sim.physics.solver_cfg.entries[1].solver_cfg
    for name in (
        "active_k",
        "anchor_particles",
        "selection_interval",
        "effective_mass_scale",
        "tangent_mass_scale",
        "horizontal_mass_scale",
        "upward_mass_scale",
        "normal_power",
        "contact_iterations",
        "contact_relaxation",
        "gradient_radius",
        "gradient_contact_band",
        "contact_stabilization",
        "contact_offset",
    ):
        assert solver[name] == getattr(g1_solver, name), name
    physics["solver_cfg"]["entries"][1]["solver_cfg"] = original_solver
    physics["solver_cfg"]["class_type"] = original_physics["solver_cfg"]["class_type"]
    physics["class_type"] = original_physics["class_type"]
    assert nerd == reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Object NeRD requires CUDA")
def test_snow_ball_contacts_and_independent_reset(monkeypatch, tmp_path):
    """Both offset worlds receive material forces; resetting one preserves the other."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Object NeRD must not construct or step native MPM")

    monkeypatch.setattr(SolverImplicitMPM, "__init__", forbidden)
    monkeypatch.setattr(SolverImplicitMPM, "step", forbidden)
    task = "IsaacContrib-ParticleNeRD-SnowBall-v0"
    cfg, _ = resolve_task_config(task, "", overrides=["env.scene.num_envs=2"])
    # A synthetic zero-output network tests the solver integration, not prediction accuracy.
    model = ParticleTransolver()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    checkpoint = tmp_path / "synthetic.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config,
            "config": {"schema": SCHEMA, "pipeline": PIPELINE, "dt": 0.005, "features": {"support": 0.07}},
        },
        checkpoint,
    )
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.checkpoint = str(checkpoint)
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.seed = 42
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            solver = NewtonManager._solver.solver("sand")
            entry = NewtonManager._solver._entries["sand"]
            actions = torch.zeros((2, env.action_manager.total_action_dim), device=env.device)
            force_max = torch.zeros(2, device=env.device)
            for step in range(100):
                observations, reward, _, _, _ = env.step(actions)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in observations.values())
                assert torch.isfinite(solver.history).all()
                force = wp.to_torch(solver.estimator.wrench)[:, :3].norm(dim=-1)
                force_max = torch.maximum(force_max, force)
                if step == 0:
                    # Scene clones differ only in world translation; learned displacement must not.
                    displacement = solver.history[:, :, -1] - solver.history[:, :, -2]
                    torch.testing.assert_close(displacement[0], displacement[1], atol=2e-5, rtol=0)
            assert (force_max > 1.0).all(), "Each object must receive material reaction forces"
            before = solver.history[1].clone()
            before_body = env.scene["snow_ball"].data.root_state_w[1].clone()
            env._reset_idx(torch.tensor([0], device=env.device))
            torch.testing.assert_close(solver.history[1], before, atol=0, rtol=0)
            torch.testing.assert_close(env.scene["snow_ball"].data.root_state_w[1], before_body, atol=0, rtol=0)
            positions = wp.to_torch(entry.state_0.particle_q).view(2, solver.capacity, 3)[0]
            torch.testing.assert_close(solver.history[0], positions[:, None].expand(-1, 3, -1))
        finally:
            env.close()
