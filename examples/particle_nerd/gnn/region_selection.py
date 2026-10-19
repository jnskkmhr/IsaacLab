# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU particle activity selection; reference-future labels are training/evaluation only."""

import numpy as np
import torch
import warp as wp
from torch import nn

from particle_nerd.gnn.inference import ParticlePredictor, advance


@wp.kernel
def neighbor_speeds(
    nodes: wp.array2d(dtype=float),
    src: wp.array(dtype=wp.int64),
    dst: wp.array(dtype=wp.int64),
    valid: wp.array(dtype=float),
    degree: wp.array(dtype=float),
    stats: wp.array2d(dtype=float),
):
    e = wp.tid()
    if valid[e] > 0.0:
        j, i = src[e], dst[e]
        speed = wp.length(wp.vec3(nodes[j, 6], nodes[j, 7], nodes[j, 8]))
        wp.atomic_add(stats, i, 0, speed / degree[i])
        wp.atomic_max(stats, i, 1, speed)


def selection_inputs(features):
    nodes, edges, boundary, influence, src, dst, degree, valid = features
    stats = torch.zeros((len(nodes), 2), device=nodes.device)
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(nodes.device))):
        wp.launch(
            neighbor_speeds,
            len(src),
            [
                wp.from_torch(nodes),
                wp.from_torch(src),
                wp.from_torch(dst),
                wp.from_torch(valid),
                wp.from_torch(degree),
                wp.from_torch(stats),
            ],
        )
    # Resolve near-rest velocities without saturating fast-motion inputs. Never
    # mutate tensors shared with the frozen motion GNN.
    scaled_nodes = nodes.clone()
    scaled_nodes[:, :9] = torch.asinh(nodes[:, :9] / 0.001)
    scaled_boundary = boundary.clone()
    scaled_boundary[:, :, 4:7] = torch.asinh(boundary[:, :, 4:7] / 0.01)
    stats = torch.log1p(stats / 0.001)
    # Body features vanish outside their local interaction radius.
    return torch.cat(
        (
            scaled_nodes,
            (scaled_boundary * influence).flatten(1),
            influence.flatten(1),
            stats,
        ),
        dim=-1,
    )


class MotionSelector(nn.Module):
    def __init__(self, hidden=64, feature_version=2):
        super().__init__()
        if feature_version != 2:
            raise ValueError("Selector checkpoint requires incompatible feature preprocessing")
        self.config = dict(hidden=hidden, feature_version=feature_version)
        self.network = nn.Sequential(
            nn.Linear(48, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


def reference_activity(f, frame, horizon=5, distance=0.0005):
    """Label motion over the next 0.1 s; never called by learned inference."""
    available = min(horizon, len(f["frame_steps"]) - frame - 1)
    if available <= 0:
        raise ValueError("A future reference frame is required for labels")
    x = f["particle_position"][frame]
    future = f["particle_position"][frame + 1 : frame + available + 1]
    displacement = np.linalg.norm(future - x[None], axis=-1).max(axis=0)
    return displacement > distance * available / horizon


@wp.kernel
def freeze_inactive(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    history: wp.array2d(dtype=float),
    old_x: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=int),
):
    i = wp.tid()
    # Preserve the invalid-state diagnostic if the graph overflowed.
    if active[i] == 0 and overflow[0] == 0:
        x[i] = old_x[i]
        v[i] = wp.vec3(0.0)
        for k in range(3):
            history[i, history.shape[1] - 3 + k] = 0.0


class SelectedPredictor(ParticlePredictor):
    """Accuracy ablation: retain full graph context, freeze inactive state updates.

    The fixed-capacity GNN still evaluates all nodes/edges in this first ablation.
    Do not interpret active fraction as measured neural computation savings.
    """

    def __init__(
        self,
        *args,
        selector_checkpoint=None,
        selector_threshold=None,
        mode="all",
        speed_threshold=0.005,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.speed_threshold = speed_threshold
        self.old_x = torch.empty_like(self.x)
        self.active = torch.ones(len(self.x), dtype=torch.int32, device=self.device)
        self.selector = None
        if selector_checkpoint is not None:
            saved = torch.load(selector_checkpoint, map_location=self.device, weights_only=False)
            if saved["config"].get("feature_version") != 2:
                raise ValueError("Retrain selector with velocity-scaled feature version 2")
            self.selector = MotionSelector(**saved["config"]).to(self.device).eval()
            self.selector.load_state_dict(saved["model"])
            self.threshold = saved["threshold"] if selector_threshold is None else selector_threshold
            self.label_horizon = saved["label_horizon"]
            self.label_distance = saved["label_distance"]
        if mode == "learned" and self.selector is None:
            raise ValueError("Learned selection requires a selector checkpoint")

    @torch.no_grad()
    def step(self, body_pose, body_velocity, active_mask=None):
        features = self.features.prepare(self.x, self.history, body_pose, body_velocity, self.material, self.mass)
        if self.mode == "oracle":
            if active_mask is None:
                raise ValueError("Oracle mode requires reference activity labels; not deployable")
            self.active.copy_(active_mask)
        elif self.mode == "learned":
            self.active.copy_(torch.sigmoid(self.selector(selection_inputs(features))) >= self.threshold)
        elif self.mode == "velocity":
            self.active.copy_(
                torch.linalg.vector_norm(self.history.reshape(-1, 3, 3), dim=-1).amax(-1) > self.speed_threshold
            )
        elif self.mode == "frozen":
            self.active.zero_()
        elif self.mode == "all":
            self.active.fill_(1)
        else:
            raise ValueError(self.mode)
        self.old_x.copy_(self.x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            correction = self.forward(*features).contiguous()
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.x.device))):
            wp.launch(
                advance,
                len(self.x),
                [
                    wp.from_torch(self.x, dtype=wp.vec3),
                    wp.from_torch(self.v, dtype=wp.vec3),
                    wp.from_torch(self.history),
                    wp.from_torch(correction),
                    self.dt,
                    self.features.overflow,
                ],
            )
            wp.launch(
                freeze_inactive,
                len(self.x),
                [
                    wp.from_torch(self.x, dtype=wp.vec3),
                    wp.from_torch(self.v, dtype=wp.vec3),
                    wp.from_torch(self.history),
                    wp.from_torch(self.old_x, dtype=wp.vec3),
                    wp.from_torch(self.active),
                    self.features.overflow,
                ],
            )
        return self.x, self.v
