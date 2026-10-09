# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-bed G1 contact, symmetry, and selected-environment reset contracts."""

import gymnasium as gym
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from tensordict import TensorDict

from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_contrib.mdp.symmetry import compute_mirrored_states

from isaaclab_tasks.contrib.g1_29dof_mpm.g1_mpm_env_cfg import G1MPMEnvCfg


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_fixed_bed_contacts_and_partial_reset(device):
    """Both rigid-platform and sand support forces reach the robot; resetting one object preserves the other."""
    cfg = G1MPMEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = device
    cfg.sim.visualizer_cfgs = []
    cfg.sim.default_visualizer_cfg = None
    cfg.video_recorders = []
    cfg.observations.policy.enable_corruption = False
    for name, term in vars(cfg.events).items():
        if getattr(term, "mode", None) in ("startup", "interval"):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    with launch_simulation(cfg.sim, launcher_args={"headless": True}):
        env = gym.make("IsaacContrib-Velocity-Sand-G1-29dof-MPM", cfg=cfg)
        try:
            raw = env.unwrapped
            obs, _ = env.reset()
            actions = torch.zeros((2, raw.action_manager.total_action_dim), device=device)
            augmented, augmented_actions = compute_mirrored_states(raw, TensorDict(obs, batch_size=[2]), actions)
            assert augmented_actions.shape == (4, raw.action_manager.total_action_dim)
            assert all(augmented[k].shape == (4, *v.shape[1:]) for k, v in obs.items())
            forces = []
            for _ in range(24):
                obs, reward, _, _, _ = env.step(actions)
                assert all(torch.isfinite(v).all() for v in obs.values())
                assert torch.isfinite(reward).all()
                forces.append(raw.foot_contact_force[..., 2].sum(dim=1).clone())
            assert (torch.stack(forces).amax(dim=0) > 100.0).all()
            model = NewtonManager.get_model()
            state = NewtonManager.get_state_0()
            other = wp.to_torch(model.particle_world) == 1
            before_position = wp.to_torch(state.particle_q)[other].clone()
            before_friction = wp.to_torch(model.mpm.friction)[other].clone()
            raw._reset_idx(torch.tensor([0], device=device))
            torch.testing.assert_close(wp.to_torch(state.particle_q)[other], before_position)
            torch.testing.assert_close(wp.to_torch(model.mpm.friction)[other], before_friction)
            assert not raw.foot_contact[0].any()
            raw.reset_sand_bed(slice(None))
            assert not raw.foot_contact.any()
            robot = raw.scene["robot"]
            ids = torch.tensor([0], device=device)
            root = robot.data.default_root_state.torch[ids].clone()
            root[:, :3] = raw.scene.env_origins[ids] + torch.tensor([0.0, 0.0, 0.76], device=device)
            robot.write_root_state_to_sim(root, env_ids=ids)
            forces = []
            for _ in range(35):
                obs, reward, _, _, _ = env.step(actions)
                assert all(torch.isfinite(v).all() for v in obs.values())
                assert torch.isfinite(reward).all()
                forces.append(wp.to_torch(raw._coupling_forces)[raw._foot_body_ids[0], 2].sum().clone())
            assert torch.stack(forces).max() > 100.0
        finally:
            env.close()
