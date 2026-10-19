# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Particle GNN with local rigid interactions and separate position/velocity heads."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def mlp(inputs, hidden, outputs, norm=True):
    layers = [nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs)]
    if norm:
        layers.append(nn.LayerNorm(outputs))
    return nn.Sequential(*layers)


class MessageBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.source = nn.Linear(hidden, hidden, bias=False)
        self.destination = nn.Linear(hidden, hidden, bias=False)
        self.edge = nn.Linear(hidden, hidden)
        self.finish = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.node = mlp(2 * hidden, hidden, hidden)

    def forward(self, node, edge, src, dst, degree, valid):
        message = self.source(node)[src] + self.destination(node)[dst] + self.edge(edge)
        edge = edge + self.finish(message)
        aggregate = torch.zeros_like(node).index_add_(0, dst, edge * valid[:, None].to(edge.dtype))
        node = node + self.node(torch.cat((node, aggregate / degree[:, None].sqrt().to(node.dtype)), dim=-1))
        return node, edge


class ParticleGNN(nn.Module):
    def __init__(
        self,
        hidden=64,
        passes=3,
        history=3,
        radius=0.07,
        dt=0.02,
        activation_checkpointing=True,
    ):
        super().__init__()
        self.config = dict(
            hidden=hidden,
            passes=passes,
            history=history,
            radius=radius,
            dt=dt,
            activation_checkpointing=activation_checkpointing,
        )
        self.node_encoder = mlp(3 * history + 17, hidden, hidden)
        self.edge_encoder = mlp(7, hidden, hidden)
        self.boundary_encoder = mlp(9, hidden, hidden)
        self.blocks = nn.ModuleList([MessageBlock(hidden) for _ in range(passes)])
        self.decoder = mlp(hidden, hidden, 6, norm=False)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, nodes, edges, boundary, influence, src, dst, degree, valid):
        node = self.node_encoder(nodes) + (self.boundary_encoder(boundary) * influence).sum(dim=1)
        edge = self.edge_encoder(edges)
        for block in self.blocks:
            if self.training and self.config["activation_checkpointing"]:
                node, edge = checkpoint(block, node, edge, src, dst, degree, valid, use_reentrant=False)
            else:
                node, edge = block(node, edge, src, dst, degree, valid)
        # Units: first three outputs correct displacement/dt [m/s], last
        # three correct endpoint velocity [m/s]. Neither predicts a force.
        return 0.2 * self.decoder(node).float()


def integrate(x, history, correction, dt):
    v = history[:, -3:]
    return x + dt * (v + correction[:, :3]), v + correction[:, 3:]
