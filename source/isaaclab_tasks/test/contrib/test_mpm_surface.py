# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU surface publication must follow material motion without stale triangles or host readback."""

import numpy as np
import pytest
import warp as wp

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.surface_cfg import SandSurfaceCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.surface import WorldSurface


@pytest.mark.kitless
def test_captured_surface_tracks_material_and_clears_unused_triangles(monkeypatch):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Captured particle surface requires CUDA")
    device = wp.get_cuda_device()
    xyz = (
        np.stack(
            np.meshgrid(
                np.arange(0.01, 0.4, 0.02),
                np.arange(0.01, 0.4, 0.02),
                np.arange(-0.19, 0.0, 0.02),
                indexing="ij",
            ),
            axis=-1,
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )
    # Reserve slots parked far away must not enlarge the visible surface or its tile map.
    xyz = np.concatenate((xyz, np.array([[1000.0, 1000.0, -1000.0]], dtype=np.float32)))
    with wp.ScopedDevice(device):
        positions = wp.array(xyz, dtype=wp.vec3)
        radii = wp.full(len(xyz), 0.01)
        ids = wp.array(np.arange(len(xyz)), dtype=wp.int32)
        cfg = SandSurfaceCfg(max_grid_cells=32768, max_vertices=16384, max_triangles=32768)
        eligible = wp.array(np.r_[np.ones(len(xyz) - 1, dtype=np.int32), np.int32(0)])
        surface = WorldSurface(cfg, device, ids, ids, radii, eligible)
        surface.update(positions, 0)

        def bounds():
            points, indices = surface.points.numpy(), surface.indices.numpy()
            # Discard degenerate padding without consulting the extractor's internal mesh counts.
            triangles = indices.reshape(-1, 3)
            live = (triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
            assert live.any()
            used = points[triangles[live].reshape(-1)]
            return np.stack((used.min(axis=0), used.max(axis=0)))

        initial_bounds = bounds()
        assert initial_bounds[1, 0] < 1.0
        offset = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        positions.assign(xyz + offset)

        def reject_readback(*args, **kwargs):
            raise AssertionError("Surface update copied a GPU array to the CPU")

        with monkeypatch.context() as patch:
            patch.setattr(wp.array, "numpy", reject_readback)
            surface.update(positions, 1)
            wp.synchronize_device(device)
        np.testing.assert_allclose(bounds(), initial_bounds + offset, atol=cfg.resolution)

        # Resetting the particle buffer must restore the visible bed through the same captured path.
        positions.assign(xyz)
        surface.update(positions, 2)
        np.testing.assert_allclose(bounds(), initial_bounds, atol=cfg.resolution)

        # An empty density field must clear old triangles even though viewer buffers never shrink.
        radii.zero_()
        surface.update(positions, 3)
        assert np.count_nonzero(surface.indices.numpy()) == 0


@pytest.mark.kitless
def test_live_surface_and_sleeping_cache_match_complete_surface():
    """A waking/sleeping region must not lose triangles or leave a seam at its cached neighbor."""
    from scipy.spatial import cKDTree

    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Captured particle surface requires CUDA")
    device = wp.get_cuda_device()
    xyz = (
        np.stack(
            np.meshgrid(
                np.arange(0.01, 1.4, 0.02), np.arange(0.01, 0.4, 0.02), np.arange(-0.19, 0.0, 0.02), indexing="ij"
            ),
            axis=-1,
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )
    with wp.ScopedDevice(device):
        q = wp.array(xyz, dtype=wp.vec3)
        r = wp.full(len(xyz), 0.01)
        ids = wp.array(np.arange(len(xyz)), dtype=int)
        activity = wp.array((xyz[:, 0] < 0.3).astype(np.int32))
        cfg = SandSurfaceCfg(max_grid_cells=65536, max_vertices=32768, max_triangles=65536)
        surface = WorldSurface(cfg, device, ids, ids, r)
        reference = WorldSurface(cfg, device, ids, ids, r)

        def geometry(meshes):
            result = []
            for points, indices, _ in meshes:
                result.append(points.numpy()[indices.numpy()].reshape(-1, 3))
            return np.concatenate(result)

        def compare():
            actual = geometry(surface.view_meshes())
            expected = geometry(reference.view_meshes())
            assert actual.shape == expected.shape
            assert cKDTree(expected).query(actual)[0].max() < 2e-5
            assert cKDTree(actual).query(expected)[0].max() < 2e-5

        surface.update(q, 0, activity)
        reference.update(q, 0)
        compare()
        assert surface.host_metadata[3] > 0  # The distant part really uses cached triangles.
        surface.update(q, 1, activity)
        surface.view_meshes()
        assert surface.host_metadata[4] == 0
        assert np.count_nonzero(surface.flags.numpy()) < len(xyz)

        # Deform only the awake end. The sleeping neighbor stays unchanged.
        deformed = xyz.copy()
        deformed[xyz[:, 0] < 0.3, 2] -= 0.03
        q.assign(deformed)
        surface.update(q, 2, activity)
        reference.update(q, 2)
        compare()

        # Finish moving, sleep, then wake the opposite end.
        activity.zero_()
        surface.update(q, 3, activity)
        surface.update(q, 4, activity)
        compare()
        assert surface.host_metadata[1] == 0
        activity.assign((xyz[:, 0] > 1.1).astype(np.int32))
        surface.update(q, 5, activity)
        compare()

        # Reset must invalidate frozen geometry, even when the native activity state is all sleeping.
        q.assign(xyz)
        activity.zero_()
        surface.reset.fill_(1)
        surface.update(q, 6, activity)
        reference.update(q, 6)
        compare()

        # Escaped particles remain visible outside the finite cache map.
        q.assign(xyz + np.array([4.0, 0.0, 0.0], dtype=np.float32))
        surface.update(q, 7, activity)
        reference.update(q, 7)
        compare()
