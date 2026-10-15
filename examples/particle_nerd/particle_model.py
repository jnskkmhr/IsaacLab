# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""All-particle graph dynamics with shared signed-distance rigid geometry inputs.

An encoder/message-passing/decoder network predicts a position correction to a
free-gravity step. Next velocity is obtained from the predicted displacement.
There is no region selector, semantic rigid-body class, or learned force output.
"""

import numpy as np
import torch
from geometry import body_contacts, body_contacts_torch
from scipy.spatial import cKDTree
from torch import nn
from torch.utils.checkpoint import checkpoint


def mlp(input_size, hidden, output_size, normalize=True):
    layers = [nn.Linear(input_size, hidden), nn.SiLU(), nn.Linear(hidden, output_size)]
    if normalize:
        layers.append(nn.LayerNorm(output_size))
    return nn.Sequential(*layers)


def message_block(node, edge, source, destination, degree, edge_update, node_update):
    edge = edge + edge_update(torch.cat((node[source], node[destination], edge), dim=-1))
    aggregate = torch.zeros_like(node).index_add_(0, destination, edge)
    node = node + node_update(torch.cat((node, aggregate / degree[:, None].sqrt()), dim=-1))
    return node, edge


class ParticleGraph(nn.Module):
    """Local graph updates every particle, including particles distant from the body."""

    def __init__(
        self,
        hidden=64,
        passes=3,
        radius=0.035,
        velocity_scale=0.1,
        history_length=3,
        input_velocity_scale=1.0,
        activation_checkpointing=False,
    ):
        super().__init__()
        self.config = dict(
            hidden=hidden,
            passes=passes,
            radius=radius,
            velocity_scale=velocity_scale,
            history_length=history_length,
            input_velocity_scale=input_velocity_scale,
            activation_checkpointing=activation_checkpointing,
        )
        self.node_encoder = mlp(3 * history_length + 7, hidden, hidden)
        self.edge_encoder = mlp(7, hidden, hidden)
        self.boundary_encoder = mlp(9, hidden, hidden)
        self.edge_updates = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(passes)])
        self.node_updates = nn.ModuleList([mlp(2 * hidden, hidden, hidden) for _ in range(passes)])
        self.decoder = mlp(hidden, hidden, 3, normalize=False)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, position, history, material, boundary, edges, dt, gravity):
        source, destination = edges
        displacement = (position[source] - position[destination]) / self.config["radius"]
        edge_features = torch.cat(
            (
                displacement,
                displacement.norm(dim=-1, keepdim=True),
                (history[source, -3:] - history[destination, -3:]) / self.config["input_velocity_scale"],
            ),
            dim=-1,
        )
        degree = torch.bincount(destination, minlength=len(position)).to(position.dtype).clamp_min(1)
        features = torch.cat((history / self.config["input_velocity_scale"], material, degree[:, None] / 32), dim=-1)
        node = self.node_encoder(features)
        # A physical distance cutoff, shared by every rigid body. The body ID
        # only associates geometry/state before this permutation-invariant sum.
        influence = (1 - boundary[..., :1].clamp_min(0)).clamp_min(0)
        boundary = torch.cat(
            (boundary[..., :4], boundary[..., 4:7] / self.config["input_velocity_scale"], boundary[..., 7:]), dim=-1
        )
        node = node + (self.boundary_encoder(boundary) * influence).sum(dim=1)
        edge = self.edge_encoder(edge_features)
        for edge_update, node_update in zip(self.edge_updates, self.node_updates, strict=True):
            arguments = (node, edge, source, destination, degree, edge_update, node_update)
            if self.config["activation_checkpointing"] and self.training and torch.is_grad_enabled():
                node, edge = checkpoint(message_block, *arguments, use_reentrant=False)
            else:
                node, edge = message_block(*arguments)
        velocity = history[:, -3:] + dt * gravity + self.config["velocity_scale"] * self.decoder(node)
        return position + dt * velocity, velocity

    @torch.no_grad()
    def predict_chunked(self, position, history, material, boundary, edges, dt, gravity, chunk_size=131072):
        """Evaluate the same complete graph with bounded temporary edge storage.

        All edges and all particles are updated in every message-passing layer.
        Edge states persist globally; chunking only limits MLP workspace. This
        inference path does not select a material region or alter graph topology.
        """
        if self.training or chunk_size < 1:
            raise ValueError("Chunked inference requires model.eval() and a positive chunk size.")
        source, destination = edges
        degree = torch.bincount(destination, minlength=len(position)).to(position.dtype).clamp_min(1)
        hidden = self.config["hidden"]
        node = position.new_empty((len(position), hidden))
        edge = position.new_empty((len(source), hidden))
        scale = self.config["input_velocity_scale"]
        for start in range(0, len(position), chunk_size):
            selection = slice(start, start + chunk_size)
            features = torch.cat(
                (history[selection] / scale, material[selection], degree[selection, None] / 32), dim=-1
            )
            body_features = boundary[selection]
            influence = (1 - body_features[..., :1].clamp_min(0)).clamp_min(0)
            body_features = torch.cat(
                (body_features[..., :4], body_features[..., 4:7] / scale, body_features[..., 7:]), dim=-1
            )
            node[selection] = self.node_encoder(features) + (self.boundary_encoder(body_features) * influence).sum(
                dim=1
            )
        for start in range(0, len(source), chunk_size):
            selection = slice(start, start + chunk_size)
            src, dst = source[selection], destination[selection]
            displacement = (position[src] - position[dst]) / self.config["radius"]
            features = torch.cat(
                (
                    displacement,
                    displacement.norm(dim=-1, keepdim=True),
                    (history[src, -3:] - history[dst, -3:]) / scale,
                ),
                dim=-1,
            )
            edge[selection] = self.edge_encoder(features)
        for edge_update, node_update in zip(self.edge_updates, self.node_updates, strict=True):
            aggregate = torch.zeros_like(node)
            for start in range(0, len(source), chunk_size):
                selection = slice(start, start + chunk_size)
                src, dst = source[selection], destination[selection]
                edge[selection] += edge_update(torch.cat((node[src], node[dst], edge[selection]), dim=-1))
                aggregate.index_add_(0, dst, edge[selection])
            # Node updates start only after every edge has used the old nodes.
            for start in range(0, len(position), chunk_size):
                selection = slice(start, start + chunk_size)
                node[selection] += node_update(
                    torch.cat((node[selection], aggregate[selection] / degree[selection, None].sqrt()), dim=-1)
                )
        velocity = torch.empty_like(position)
        for start in range(0, len(position), chunk_size):
            selection = slice(start, start + chunk_size)
            velocity[selection] = (
                history[selection, -3:] + dt * gravity + self.config["velocity_scale"] * self.decoder(node[selection])
            )
        return position + dt * velocity, velocity


def graph_inputs(position, velocity_history, episode, body_pose, body_velocity, device, radius=0.035, sdf=None):
    """Construct features from available particle and advanced rigid states.

    In the lagged coupler the rigid transition precedes the particle transition.
    Training uses that recorded advanced rigid state; deployment computes it
    from the previous estimated wrench. No reference force enters these features.
    """
    position = torch.as_tensor(position, dtype=torch.float32, device=device)
    history = torch.as_tensor(velocity_history, dtype=torch.float32, device=device)
    pairs = cKDTree(position.detach().cpu().numpy()).query_pairs(radius, output_type="ndarray")
    edge_array = np.concatenate((pairs.T, pairs[:, ::-1].T), axis=1)
    boundary = []
    for i, pose in enumerate(body_pose):
        if sdf is not None and i == episode["moving_body"]:
            values = body_contacts(
                position.detach().cpu().numpy(), pose, episode["body_size"][i], episode["body_shapes"][i], sdf
            )
            distance, normal, closest = [torch.as_tensor(a, dtype=position.dtype, device=device) for a in values]
        else:
            distance, normal, closest = body_contacts_torch(
                position, pose, episode["body_size"][i], episode["body_shapes"][i]
            )
        state = torch.as_tensor(body_velocity[i], dtype=position.dtype, device=device)
        origin = torch.as_tensor(pose[:3], dtype=position.dtype, device=device)
        surface_velocity = state[:3] + torch.linalg.cross(state[3:].expand_as(closest), closest - origin)
        inverse_mass = 1 / episode["body_mass"][i] if episode["body_mass"][i] > 0 else 0
        boundary.append(
            torch.cat(
                (
                    (distance / 0.1).clamp(-2, 2)[:, None],
                    normal,
                    surface_velocity - history[:, -3:],
                    torch.full((len(position), 1), inverse_mass, device=device),
                    torch.full((len(position), 1), float(episode["body_friction"][i]), device=device),
                ),
                dim=-1,
            )
        )
    m = episode["material"]
    material = np.column_stack(
        (
            np.full(len(position), m["density"] / 1600),
            np.full(len(position), m["friction"]),
            np.full(len(position), np.log10(m["young_modulus"]) / 15),
            np.full(len(position), m["poisson_ratio"]),
            episode["particle_radius"] / 0.01,
            episode["particle_mass"] / 0.01,
        )
    )
    return [
        position,
        history,
        torch.as_tensor(material, dtype=torch.float32, device=device),
        torch.stack(boundary, dim=1),
        torch.as_tensor(edge_array, dtype=torch.long, device=device),
        episode["dt"],
        torch.as_tensor(episode["gravity"], dtype=torch.float32, device=device),
    ]
