# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cached MPM history survives robot/object interaction, travel, and selective reset."""

from unittest.mock import patch as mock_patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.assets import RigidObjectCfg
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401 -- register the actual task
from isaaclab_tasks.contrib.velocity_mpm.config.deprecated.g1_29dof_mpm_moving_patch_particle_cache.env_cfg.terrain_cfg import (  # noqa: E501
    FLAT_TERRAINS_CFG,
)
from isaaclab_tasks.contrib.velocity_mpm.config.deprecated.g1_29dof_mpm_moving_patch_particle_cache.util import (
    settled_template,  # noqa: E501
)
from isaaclab_tasks.contrib.velocity_mpm.config.deprecated.g1_29dof_mpm_moving_patch_particle_cache.util.particle_sampling import (  # noqa: E501
    CachedMPMGridCfg,
    _sample_grid_tile,
)
from isaaclab_tasks.contrib.velocity_mpm.config.deprecated.g1_29dof_mpm_moving_patch_particle_cache.util.surface import (  # noqa: E501
    ParticleCacheSurface,
)
from isaaclab_tasks.utils import resolve_task_config


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_particle_history_with_independent_rigid_objects(device):
    """Preserve constitutive state across paging, with two-way object contact and isolated resets."""
    task = "IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-ParticleCache"
    cfg, _ = resolve_task_config(
        task, "rsl_rl_cfg_entry_point", overrides=["env.scene.num_envs=2", f"env.sim.device={device}"]
    )
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG
    for name in vars(cfg.curriculum):
        if not name.startswith("_"):
            setattr(cfg.curriculum, name, None)
    for name in vars(cfg.terminations):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    t = cfg.scene.terrain.moving_patch_terrain
    t.voxel_size = 0.08
    t.simulated_terrain_size = (0.4, 0.4)
    t.boundary_terrain_size = 0.2
    t.particle_depth = 0.16
    t.tracked_body_patterns = [r"/World/envs/env_[^/]+/Box"]
    t.particle_cache.simulation_capacity_multiplier = 4
    t.particle_cache.max_cached_particles_per_env = 10000
    t.particle_cache.max_visited_tiles_per_env = 1024
    cfg.scene.box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.0, 0.4)),
    )
    cfg.events.mpm_material.params["parameter_ranges"] = {"density": (1600.0, 1600.0), "friction": (0.48, 0.48)}
    cfg.events.mpm_material_log = None
    cfg.events.push_robot = None
    cfg.events.reset_base.params = {"pose_range": {}, "velocity_range": {}}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    with (
        mock_patch.object(settled_template, "settle_template", wraps=settled_template.settle_template) as settle,
        launch_simulation(cfg),
    ):
        raw = gym.make(task, cfg=cfg)
        env = raw.unwrapped
        # GL/RTX own particle drawing; an authored cloud would bypass their GUI toggles.
        for prim in sim_utils.get_current_stage().Traverse():
            if "/Sand/" in str(prim.GetPath()) and prim.IsA(UsdGeom.Points):
                assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
        env.reset()
        patch = env.moving_patch_particle
        cache = patch.cache
        model = patch.model
        backend = env.sim.backend_manager.backend
        ids = [i for i, label in enumerate(model.body_label) if label.endswith("/Box")]
        assert len(ids) == 2, model.body_label
        assert (patch.body_counts == 2).all(), patch.body_counts
        maxforce = np.zeros(2)
        initial = backend.state_0.body_q.numpy()[ids, :3].copy()
        for _ in range(100):
            env.step(torch.zeros((2, env.action_manager.total_action_dim), device=env.device))
            forces = env._resolve_coupling_feedback_buffer().numpy()[ids, :3]
            assert np.isfinite(forces).all()
            maxforce = np.maximum(maxforce, np.linalg.norm(forces, axis=1))
        final = backend.state_0.body_q.numpy()[ids, :3]
        assert (final[:, 2] < initial[:, 2] - 0.1).all()
        assert (maxforce > 1.0).all()
        assert (final[:, 2] > -0.15).all()

        # Executed inside the two-world integration scene, after real contacts have deformed the bed.
        def canonical(world, tile_set=None):
            layers = cache.layers.numpy()
            keep = (cache.worlds == world) & (layers != 0)
            out = {}
            for name in ("q", "qd", "grad", "elastic", "plastic", "stress", "transform", "mass", "radius", "flags"):
                data = getattr(cache.live, name).numpy()[keep].copy()
                frozen = getattr(cache.frozen, name).numpy()[keep]
                data[layers[keep] == 2] = frozen[layers[keep] == 2]
                out[name] = data
            q = out["q"]
            keys = np.floor(q[:, :2] / cache.cfg.tile_size).astype(int)
            selected = (
                np.ones(len(q), dtype=bool) if tile_set is None else np.array([tuple(k) in tile_set for k in keys])
            )
            order = np.lexsort(q[selected].T[::-1])
            return {name: data[selected][order] for name, data in out.items()}

        class MeshRecorder:
            def __init__(self):
                self.uploads = {}
                self.visible = {}
                self.particles = {}
                self.show_particles = True
                self.show_isosurface = False

            def log_points(self, name, points, **kwargs):
                self.particles[name] = np.empty((0, 3)) if kwargs.get("hidden", False) else points.numpy().copy()

            def log_mesh(self, name, points, indices, normals=None, **kwargs):
                self.uploads[name] = self.uploads.get(name, 0) + 1
                self.visible[name] = not kwargs.get("hidden", False)
                if len(indices):
                    assert np.isfinite(points.numpy()).all()
                    assert indices.numpy().min() >= 0 and indices.numpy().max() < len(points)
                if name.startswith("/particle_cache/"):
                    assert kwargs["roughness"] == 1.0
                if name == "/particle_cache/background":
                    # The background must leave openings only for rendered loaded patches.
                    triangles = points.numpy()[indices.numpy().reshape(-1, 3)]
                    centers = np.floor(triangles.mean(axis=1)[:, :2] / cache.cfg.tile_size).astype(int)
                    self.background_tiles = set(map(tuple, centers))

        viewer = MeshRecorder()
        surface = ParticleCacheSurface(patch, backend)
        state = backend.state_0
        original_bodies = state.body_q.numpy().copy()
        original_tiles = set(cache.loaded[0])
        before = canonical(0, original_tiles)
        assert np.max(np.abs(before["stress"])) > 0
        moved = original_bodies.copy()
        moved[model.body_world.numpy() == 0, 0] += 3.0
        state.body_q.assign(moved)
        patch.update(state)
        assert cache.cached_particle_counts[0] > 0
        surface.update(viewer, state, [0], show_surface=False)
        assert (cache.loaded[0] | original_tiles).issubset(viewer.background_tiles)
        surface.update(viewer, state, [0])
        surface_name = "/particle_cache/world_0/surface"
        assert viewer.visible[surface_name]
        assert viewer.background_tiles.isdisjoint(cache.loaded[0])
        assert original_tiles.issubset(viewer.background_tiles)
        assert set(viewer.uploads) == {"/particle_cache/background", surface_name}
        points, indices, _ = surface.meshes[0]
        triangles = points.numpy()[indices.numpy().reshape(-1, 3)]
        centers = np.floor(triangles.mean(axis=1)[:, :2] / cache.cfg.tile_size).astype(int)
        # Skirt faces lie exactly on tile edges, shared with unloaded neighbors.
        xy = triangles.mean(axis=1)[:, :2]
        lower = np.asarray(sorted(cache.loaded[0])) * cache.cfg.tile_size
        inside = (xy[:, None] >= lower - 1.0e-5) & (xy[:, None] <= lower + cache.cfg.tile_size + 1.0e-5)
        assert inside.all(axis=2).any(axis=1).all()
        assert not any(tuple(center) in original_tiles for center in centers)
        state.body_q.assign(original_bodies)
        patch.update(state)
        surface.update(viewer, state, [0])
        assert set(viewer.uploads) == {"/particle_cache/background", surface_name}
        assert viewer.background_tiles.isdisjoint(cache.loaded[0])
        surface.update(viewer, state, [0], show_surface=False)
        assert not viewer.visible[surface_name]
        assert cache.loaded[0].issubset(viewer.background_tiles)
        surface.update(viewer, state, [0])
        assert viewer.visible[surface_name]
        assert viewer.background_tiles.isdisjoint(cache.loaded[0])
        after = canonical(0, original_tiles)
        for name in before:
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)
        assert len(before["q"]) > 0
        untouched = canonical(0)
        event = env.event_manager.get_term_cfg("mpm_material")
        event.params["parameter_ranges"] = {"density": (2000.0, 2000.0), "friction": (0.8, 0.8)}
        env.event_manager.set_term_cfg("mpm_material", event)
        env.reset(env_ids=torch.tensor([1], device=env.device))
        patch.update(state)
        for name, data in untouched.items():
            np.testing.assert_array_equal(canonical(0)[name], data, err_msg="partial reset " + name)
        assert not cache.pages[1]
        # Material randomization must not launch another offline settling simulation.
        assert settle.call_count == 1

        surface.update(viewer, state, [0])
        points, indices, normals = surface.meshes[0]
        assert len(points) > 0 and len(indices) > 0
        assert np.isfinite(points.numpy()).all() and np.isfinite(normals.numpy()).all()
        assert indices.numpy().min() >= 0 and indices.numpy().max() < len(points)

        # A reset after capture must affect the next solver step. Material changes
        # stay isolated to world 1, and new joint targets must still reach physics.
        samples = env.event_manager.get_term_cfg("mpm_material").func.material_parameters
        torch.testing.assert_close(samples["density"], torch.tensor([1600.0, 2000.0], device=env.device))
        joint_before = env.scene["robot"].data.joint_pos.torch.clone()
        for _ in range(10):
            action = torch.full((2, env.action_manager.total_action_dim), 0.2, device=env.device)
            _, reward, _, _, _ = env.step(action)
            assert torch.isfinite(reward).all()
            assert torch.isfinite(env.scene["robot"].data.root_pos_w.torch).all()
        assert not torch.allclose(env.scene["robot"].data.joint_pos.torch, joint_before)
        # Reset keeps the surface local.
        env.reset(env_ids=torch.tensor([0], device=env.device))
        patch.update(backend.state_0)
        surface.update(viewer, backend.state_0, [0])
        assert set(viewer.uploads) == {"/particle_cache/background", surface_name}


@pytest.mark.parametrize(("multiplier", "tiles"), [(1, 64), (1.01, 65), (1.25, 80), (2.0, 128)])
def test_fractional_slot_multiplier_rounds_up_to_whole_tiles(multiplier, tiles):
    """Reserve slots come in whole template tiles, so a fractional multiplier rounds up to the next tile."""
    cfg = CachedMPMGridCfg(
        lower=(-0.8, -0.8, -0.25),
        upper=(0.8, 0.8, 0.0),
        voxel_size=0.04,
        particles_per_cell=1.25,
        particle_placement="cell_center",
        jitter=0.05,
        tile_size=0.2,
        simulation_capacity_multiplier=multiplier,
    )
    template, _, _ = _sample_grid_tile(cfg)
    assert cfg.get_particle_count() == len(template) * tiles
