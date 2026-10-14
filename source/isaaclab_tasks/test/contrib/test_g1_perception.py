# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coordinate, masking, and exported-policy contracts for G1 perception."""

import math
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.agents.pointnet import PointNetMLPModel
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.depth import depth_image
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.observations import lidar_point_cloud
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.stair_rewards import (
    feet_stumble,
    local_base_height_l2,
    stair_foot_clearance,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.symmetry import mirror_elevation_map


def test_elevation_map_mirror_moves_terrain_to_the_opposite_lateral_cell():
    """A raised left-hand cell becomes a raised right-hand cell with valid grid coordinates."""
    data = torch.tensor(
        [
            [
                [[0.0, -1.0, -0.8], [0.0, 0.0, -0.6], [0.0, 1.0, -0.4]],
                [[1.0, -1.0, -2.0], [1.0, 0.0, -0.6], [1.0, 1.0, -0.3]],
            ]
        ]
    )
    original = data.clone()
    mirrored = mirror_elevation_map(data)
    torch.testing.assert_close(mirrored[..., :2], data[..., :2])
    torch.testing.assert_close(mirrored[..., 2], torch.tensor([[[-0.4, -0.6, -0.8], [-0.3, -0.6, -2.0]]]))
    torch.testing.assert_close(mirror_elevation_map(mirrored), original)
    torch.testing.assert_close(data, original)


def test_stumble_penalty_distinguishes_stair_impacts_from_support_and_noise():
    """Horizontal impact on either foot is penalized; support, no contact and tiny noise are not."""
    forces = torch.tensor(
        [
            [[0.0, 0.0, 200.0], [0.0, 0.0, 0.0]],
            [[0.0, 60.0, 10.0], [0.0, 0.0, 200.0]],
            [[0.0, 0.0, 200.0], [-60.0, 0.0, -10.0]],
            [[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    sensor = SimpleNamespace(data=SimpleNamespace(net_normal_forces_w=SimpleNamespace(torch=forces)))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contacts": sensor}))
    cfg = SceneEntityCfg("contacts", body_ids=[0, 1])
    torch.testing.assert_close(feet_stumble(env, cfg), torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_stair_rewards_use_local_treads_and_do_not_reward_motionless_feet():
    """Adjacent stair heights must not bias local clearance; standing cannot earn swing reward."""
    velocity = torch.zeros(1, 2, 3)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor([[1.0, 0.0, 0.95]])),
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.14], [1.0, 0.0, 0.34]]])),
            body_lin_vel_w=SimpleNamespace(torch=velocity),
        )
    )
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            ray_hits_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [math.nan] * 3]]))
        )
    )
    command = torch.tensor([[0.75, 0.0, 0.0]])
    env = SimpleNamespace(
        scene={"robot": robot, "scanner": scanner}, command_manager=SimpleNamespace(get_command=lambda name: command)
    )
    sensor_cfg = SceneEntityCfg("scanner")
    torch.testing.assert_close(local_base_height_l2(env, 0.75, sensor_cfg), torch.zeros(1))
    params = dict(
        target_height=0.1,
        std=0.05,
        tanh_mult=2.0,
        asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1]),
        height_sensor_cfg=sensor_cfg,
        standing_position_foot_z=0.04,
    )
    torch.testing.assert_close(stair_foot_clearance(env, **params), torch.zeros(1))
    velocity[..., 0] = 10.0
    torch.testing.assert_close(stair_foot_clearance(env, **params), torch.ones(1))
    command.zero_()
    torch.testing.assert_close(stair_foot_clearance(env, **params), torch.zeros(1))


def test_lidar_uses_measurement_frame_and_masks_misses():
    """World translation/yaw must not leak into the sensor-frame cloud; misses stay finite."""
    sensor = SimpleNamespace(
        num_rays=5,
        cfg=SimpleNamespace(ray_alignment="base", max_distance=40.0),
        data=SimpleNamespace(
            pos_w=SimpleNamespace(torch=torch.tensor([[10.0, 20.0, 1.0]])),
            quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])),
            ray_hits_w=SimpleNamespace(
                torch=torch.tensor(
                    [[[10.0, 22.0, 1.0], [9.0, 20.0, 1.0], [math.inf] * 3, [math.nan] * 3, [10.0, 61.0, 1.0]]]
                )
            ),
        ),
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"lidar": sensor}), device="cpu")
    cfg = ObservationTermCfg(func=lidar_point_cloud, params={"sensor_cfg": SceneEntityCfg("lidar"), "num_points": 5})
    term = lidar_point_cloud(cfg, env)
    actual = term(env, **cfg.params)
    expected = torch.tensor([[[2.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0] * 4, [0.0] * 4, [0.0] * 4]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def _make_model():
    torch.manual_seed(4)
    points = torch.randn(2, 8, 4)
    points[..., 3] = 1.0
    points[:, -2:, 3] = 0.0
    obs = TensorDict({"policy": torch.randn(2, 6), "lidar": points, "command": torch.randn(2, 3)}, batch_size=[2])
    model = PointNetMLPModel(
        obs,
        {"actor": ["policy", "lidar", "command"]},
        "actor",
        3,
        hidden_dims=(16,),
        pointnet_hidden_dims=(8,),
        pointnet_feature_dim=8,
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    return model, obs


def test_policy_is_point_order_invariant_and_ignores_padding():
    """Point normalization/pooling must not learn point order or treat misses as surfaces."""
    model, obs = _make_model()
    model.update_normalization(obs)
    model.eval()
    expected = model(obs)
    padded = obs.clone()
    padded["lidar"][:, -2:, :3] = math.nan
    padded["lidar"] = padded["lidar"][:, torch.randperm(8)]
    torch.testing.assert_close(model(padded), expected)
    empty = obs.clone()
    empty["lidar"][..., :3] = math.nan
    empty["lidar"][..., 3] = 0.0
    assert torch.isfinite(model(empty)).all()
    model.zero_grad()
    model(obs).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.pointnet.parameters())
    assert any(torch.count_nonzero(p.grad) > 0 for p in model.pointnet.parameters())


def test_policy_checkpoint_and_exports_preserve_point_encoder(tmp_path):
    """Saving/exporting must include the encoder and trained normalization, not only the MLP head."""
    model, obs = _make_model()
    model.update_normalization(obs)
    model.eval()
    expected = model(obs)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored, _ = _make_model()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(obs), expected)
    inputs = (torch.cat((obs["policy"], obs["command"]), dim=-1), obs["lidar"])
    scripted = torch.jit.script(restored.as_jit())
    scripted.save(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(*inputs), expected)
    exported = restored.as_onnx()
    torch.testing.assert_close(exported(*inputs), expected)
    # The RSL-RL exporter consumes these same two inputs.
    assert tuple(x.shape for x in exported.get_dummy_inputs()) == ((1, 9), (1, 8, 4))


def test_depth_crop_units_misses_and_measurement_cache():
    """Depth stays metric before normalization; repeated reads and partial resets preserve frame ownership."""
    raw = torch.full((2, 3, 7, 1), 100.0)
    raw[:, 1, 1:6, 0] = torch.tensor([0.3, 1.65, 3.0, 0.0, math.nan])
    sensor = SimpleNamespace(
        data=SimpleNamespace(output={"depth": SimpleNamespace(torch=raw)}),
        frame=SimpleNamespace(torch=torch.ones(2, dtype=torch.long)),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"camera": sensor}), device="cpu", num_envs=2, common_step_counter=0
    )
    cfg = ObservationTermCfg(
        func=depth_image,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "resize": (1, 5),
            "crop": (1, 1, 1, 1),
            "corrupt": False,
        },
    )
    term = depth_image(cfg, env)
    expected = torch.tensor([[[[-0.5, 0.0, 0.5, 0.5, 0.5]]]]).expand(2, -1, -1, -1)
    torch.testing.assert_close(term(env, **cfg.params), expected)
    assert torch.isnan(raw[:, 1, 5, 0]).all()  # Sensor storage must not be modified.
    raw.fill_(3.0)
    torch.testing.assert_close(term(env, **cfg.params), expected)
    env.common_step_counter += 1
    torch.testing.assert_close(term(env, **cfg.params), expected)  # No new camera frame yet.
    sensor.frame.torch[0] += 1
    env.common_step_counter += 1
    updated = term(env, **cfg.params).clone()
    torch.testing.assert_close(updated[0], torch.full((1, 1, 5), 0.5))
    torch.testing.assert_close(updated[1], expected[1])
    # A reset can occur without advancing the global step or changing the camera's frame number.
    raw[0].fill_(0.3)
    term.reset(torch.tensor([0]))
    reset = term(env, **cfg.params)
    torch.testing.assert_close(reset[0], torch.full((1, 1, 5), -0.5))
    torch.testing.assert_close(reset[1], updated[1])


def test_elevation_map_preserves_yaw_frame_grid_and_missing_cells():
    """A translated/pitched scanner keeps XYZ relative to its origin and finite missing cells."""
    from isaaclab.sensors import patterns

    from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.env_cfg.sensors_cfg import elevation_grid_pattern
    from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.observations import elevation_map

    half_pitch = math.pi / 12
    half_yaw = math.pi / 4
    quat = torch.tensor(
        [
            [
                -math.sin(half_pitch) * math.sin(half_yaw),
                math.sin(half_pitch) * math.cos(half_yaw),
                math.cos(half_pitch) * math.sin(half_yaw),
                math.cos(half_pitch) * math.cos(half_yaw),
            ]
        ]
    )
    sensor = SimpleNamespace(
        num_rays=6,
        cfg=SimpleNamespace(
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                func=elevation_grid_pattern, resolution=1.0, size=(1.0, 2.0), ordering="yx"
            ),
        ),
        data=SimpleNamespace(
            pos_w=SimpleNamespace(torch=torch.tensor([[10.0, 20.0, 1.0]])),
            quat_w=SimpleNamespace(torch=quat),
            ray_hits_w=SimpleNamespace(
                torch=torch.tensor(
                    [
                        [
                            [11.0, 19.5, 0.5],
                            [10.0, 19.5, 0.6],
                            [9.0, 19.5, 0.7],
                            [11.0, 20.5, 0.8],
                            [math.nan] * 3,
                            [math.inf] * 3,
                        ]
                    ]
                )
            ),
        ),
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"map": sensor}), device="cpu")
    cfg = ObservationTermCfg(func=elevation_map, params={"sensor_cfg": SceneEntityCfg("map"), "missing_height": -2.0})
    term = elevation_map(cfg, env)
    expected = torch.tensor(
        [
            [
                [[-0.5, -1.0, -0.5], [-0.5, 0.0, -0.4], [-0.5, 1.0, -0.3]],
                [[0.5, -1.0, -0.2], [0.5, 0.0, -2.0], [0.5, 1.0, -2.0]],
            ]
        ]
    )
    torch.testing.assert_close(term(env, **cfg.params), expected, atol=1e-6, rtol=1e-6)
    assert torch.isnan(sensor.data.ray_hits_w.torch[0, 4]).all()
