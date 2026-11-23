# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous convolution over the neighbourhood of every particle."""

from __future__ import annotations

import torch
import warp as wp
from torch import nn

from ..nerd_kernels import collider_basis, neighbor_basis_and_packing, warp_stream


class LocalNeighborhoodAggregator(nn.Module):
    """Continuous 3D convolution (4 x 4 x 4 cells) over neighbouring particles and collider surface points.

    The Warp kernels build fixed bases (not differentiated); the two linear layers are the learned filters. The
    particle basis comes with a 6D packing state: neighbour count, packing sum, its rate, mean relative velocity.
    """

    def __init__(
        self,
        fluid_in_channels: int = 4,
        obs_in_channels: int = 9,
        filters: int = 64,
        radius: float = 0.072,
        max_neighbors: int = 48,
    ):
        super().__init__()
        self.fluid_in_channels = fluid_in_channels
        self.obs_in_channels = obs_in_channels
        self.filters = filters
        self.radius = float(radius)
        self.max_neighbors = int(max_neighbors)
        num_cells = 64

        self.fluid_linear = nn.Linear(num_cells * fluid_in_channels, filters, bias=True)
        self.obs_linear = nn.Linear(num_cells * obs_in_channels, filters, bias=True)
        nn.init.uniform_(self.fluid_linear.weight, -0.02, 0.02)
        nn.init.zeros_(self.fluid_linear.bias)
        nn.init.uniform_(self.obs_linear.weight, -0.02, 0.02)
        nn.init.zeros_(self.obs_linear.bias)

        self._grids: dict[tuple[str, int, bool], wp.HashGrid] = {}
        self._group_arrays: dict[tuple[str, int, int], wp.array] = {}
        self._eval_buffers: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get_grid(self, device: str, n: int, with_groups: bool = False) -> wp.HashGrid:
        """Hash grid reserved for ``n`` points (cached per device and size)."""
        key = (device, n, with_groups)
        grid = self._grids.get(key)
        if grid is None:
            grid = wp.HashGrid(32, 32, 16, device=device)
            grid.reserve(n, with_groups=with_groups)
            self._grids[key] = grid
        return grid

    def _get_groups(self, device: str, num_envs: int, m_pts: int, torch_device: torch.device) -> wp.array:
        key = (device, num_envs, m_pts)
        grp = self._group_arrays.get(key)
        if grp is None:
            env_ids = torch.arange(num_envs, device=torch_device, dtype=torch.int32).repeat_interleave(m_pts)
            grp = wp.from_torch(env_ids, dtype=wp.int32, requires_grad=False)
            self._group_arrays[key] = grp
        return grp

    @torch.no_grad()
    def compute_bases(
        self,
        pos_pred: torch.Tensor,
        vel_pred: torch.Tensor,
        pairs: torch.Tensor,
        influence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Particle basis [N, 64 * 4], collider basis [N, 64 * 9] and packing state [N, 6] of one environment."""
        n = pos_pred.shape[0]
        device = str(pos_pred.device)
        pos_f = pos_pred.detach().float()
        pos_c = (pos_f - pos_f.mean(dim=0, keepdim=True)).contiguous()
        vel_c = vel_pred.detach().float().contiguous()
        pairs_c = pairs.detach().float().contiguous()
        inf_c = influence.detach().float().contiguous()

        if not self.training:
            buf_key = (device, n, torch.is_inference_mode_enabled())
            bufs = self._eval_buffers.get(buf_key)
            if bufs is None:
                fluid_basis = torch.zeros((n, 64, self.fluid_in_channels), dtype=torch.float32, device=pos_pred.device)
                obs_basis = torch.zeros((n, 64, self.obs_in_channels), dtype=torch.float32, device=pos_pred.device)
                packing = torch.zeros((n, 6), dtype=torch.float32, device=pos_pred.device)
                self._eval_buffers[buf_key] = (fluid_basis, obs_basis, packing)
            else:
                fluid_basis, obs_basis, packing = bufs
                fluid_basis.zero_()
                obs_basis.zero_()
                packing.zero_()
        else:
            fluid_basis = torch.zeros((n, 64, self.fluid_in_channels), dtype=torch.float32, device=pos_pred.device)
            obs_basis = torch.zeros((n, 64, self.obs_in_channels), dtype=torch.float32, device=pos_pred.device)
            packing = torch.zeros((n, 6), dtype=torch.float32, device=pos_pred.device)

        grid = self.get_grid(device, n)
        pos_wp = wp.from_torch(pos_c, dtype=wp.vec3, requires_grad=False)
        vel_wp = wp.from_torch(vel_c, dtype=wp.vec3, requires_grad=False)
        fluid_wp = wp.from_torch(fluid_basis, requires_grad=False)
        obs_wp = wp.from_torch(obs_basis, requires_grad=False)
        pack_wp = wp.from_torch(packing, requires_grad=False)

        with wp.ScopedStream(warp_stream(pos_pred.device)):
            grid.build(pos_wp, self.radius)
            wp.launch(
                neighbor_basis_and_packing,
                n,
                [grid.id, pos_wp, vel_wp, self.radius, self.max_neighbors, 0, fluid_wp, pack_wp],
                device=device,
            )
            if pairs_c.shape[1] > 0:
                wp.launch(
                    collider_basis,
                    n,
                    [
                        wp.from_torch(pairs_c, requires_grad=False),
                        wp.from_torch(inf_c, requires_grad=False),
                        obs_wp,
                    ],
                    device=device,
                )
        return fluid_basis.view(n, -1), obs_basis.view(n, -1), packing

    @torch.no_grad()
    def compute_bases_batched(
        self,
        pos_pred_env: torch.Tensor,
        vel_pred_env: torch.Tensor,
        pairs_env: torch.Tensor,
        influence_env: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:meth:`compute_bases` for [E, M, 3] particles in one launch; neighbours stay within their environment."""
        num_envs, m_pts, _ = pos_pred_env.shape
        n_tot = num_envs * m_pts
        n_col = pairs_env.shape[2]
        device = str(pos_pred_env.device)

        pos_f = pos_pred_env.detach().float()
        pos_c = (pos_f - pos_f.mean(dim=1, keepdim=True)).reshape(n_tot, 3).contiguous()
        vel_c = vel_pred_env.detach().float().reshape(n_tot, 3).contiguous()
        pairs_c = pairs_env.detach().float().reshape(n_tot, n_col, 8).contiguous()
        inf_c = influence_env.detach().float().reshape(n_tot, n_col).contiguous()

        buf_key = (device, n_tot, torch.is_inference_mode_enabled())
        bufs = self._eval_buffers.get(buf_key)
        if bufs is None:
            fluid_basis = torch.zeros(
                (n_tot, 64, self.fluid_in_channels), dtype=torch.float32, device=pos_pred_env.device
            )
            obs_basis = torch.zeros((n_tot, 64, self.obs_in_channels), dtype=torch.float32, device=pos_pred_env.device)
            packing = torch.zeros((n_tot, 6), dtype=torch.float32, device=pos_pred_env.device)
            self._eval_buffers[buf_key] = (fluid_basis, obs_basis, packing)
        else:
            fluid_basis, obs_basis, packing = bufs
            fluid_basis.zero_()
            obs_basis.zero_()
            packing.zero_()

        grid = self.get_grid(device, n_tot, with_groups=True)
        groups_wp = self._get_groups(device, num_envs, m_pts, pos_pred_env.device)
        pos_wp = wp.from_torch(pos_c, dtype=wp.vec3, requires_grad=False)
        vel_wp = wp.from_torch(vel_c, dtype=wp.vec3, requires_grad=False)
        fluid_wp = wp.from_torch(fluid_basis, requires_grad=False)
        obs_wp = wp.from_torch(obs_basis, requires_grad=False)
        pack_wp = wp.from_torch(packing, requires_grad=False)

        with wp.ScopedStream(warp_stream(pos_pred_env.device)):
            grid.build(pos_wp, self.radius, groups=groups_wp)
            wp.launch(
                neighbor_basis_and_packing,
                n_tot,
                [grid.id, pos_wp, vel_wp, self.radius, self.max_neighbors, m_pts, fluid_wp, pack_wp],
                device=device,
            )
            if n_col > 0:
                wp.launch(
                    collider_basis,
                    n_tot,
                    [
                        wp.from_torch(pairs_c, requires_grad=False),
                        wp.from_torch(inf_c, requires_grad=False),
                        obs_wp,
                    ],
                    device=device,
                )
        return (
            fluid_basis.view(num_envs, m_pts, -1),
            obs_basis.view(num_envs, m_pts, -1),
            packing.view(num_envs, m_pts, 6),
        )

    def forward(
        self,
        fluid_basis_flat: torch.Tensor,
        obs_basis_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.fluid_linear.weight.dtype
        return self.fluid_linear(fluid_basis_flat.to(dt)), self.obs_linear(obs_basis_flat.to(dt))
