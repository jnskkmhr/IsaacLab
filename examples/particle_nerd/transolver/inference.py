# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU particle displacement predictor using ParticleTransolver."""

import torch
import warp as wp

from particle_nerd.transolver.features import features
from particle_nerd.transolver.model import ParticleTransolver


@wp.kernel
def update(
    history: wp.array3d(dtype=float),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    displacement: wp.array2d(dtype=float),
    active: wp.array(dtype=int),
    dt: float,
):
    i = wp.tid()
    delta = wp.vec3(0.0)
    if active[i] != 0:
        delta = wp.vec3(displacement[i, 0], displacement[i, 1], displacement[i, 2])
    next_x = x[i] + delta
    x[i] = next_x
    v[i] = delta / dt
    for t in range(history.shape[1] - 1):
        for c in range(3):
            history[i, t, c] = history[i, t + 1, c]
    for c in range(3):
        history[i, history.shape[1] - 1, c] = next_x[c]


def box_ground_colliders(size, device, friction=0.6):
    """Adapter for the single-box + ground dataset."""
    return (
        torch.tensor([1, 0], dtype=torch.int32, device=device),
        torch.tensor([[s / 2 for s in size], [0, 0, 0]], dtype=torch.float32, device=device),
        torch.full((2,), friction, dtype=torch.float32, device=device),
    )


def manifest_colliders(manifest, device):
    """Material particles query each rigid collider; shovel uses its concave SDF."""
    descriptors = box_ground_colliders(manifest["body_size_m"], device, manifest.get("body_contact_friction", 0.6))
    if manifest.get("geometry") == "shovel":
        descriptors[0][0] = 3
        descriptors[2][1] = 0.6
    return descriptors


def manifest_sdf(manifest, device):
    from particle_nerd.shovel.sdf import tensors

    return tensors(str(device), manifest.get("geometry", "none"))


def box_ground_state(pose, velocity):
    ground = pose.new_tensor([0, 0, 0, 0, 0, 0, 1])
    return torch.stack((pose, ground)), torch.stack((velocity, torch.zeros_like(velocity)))


class Predictor:
    def __init__(
        self,
        saved,
        history,
        material,
        kinds=None,
        sizes=None,
        friction=None,
        geometry=None,
        use_cuda_graph=True,
        active_k=None,
        anchor_m=64,
    ):
        arch = saved.get("architecture")
        if arch != "particle_transolver_v1":
            raise ValueError(f"Expected particle_transolver_v1, got {arch}")
        wp.init()
        self.architecture = arch
        self.config = saved["config"]
        self.device, self.dt = str(history.device), self.config["dt"]
        self.model = ParticleTransolver(**self.config["model"]).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        if history.shape[1] != self.config["model"]["history"]:
            raise ValueError("History length differs from checkpoint")
        if kinds is None:
            if sizes is not None or friction is not None:
                raise ValueError("Supply all collider descriptors together")
            kinds, sizes, friction = manifest_colliders(self.config, self.device)
        elif sizes is None or friction is None:
            raise ValueError("Supply kinds, sizes, and friction together")
        self.sdf = manifest_sdf({"geometry": geometry or self.config.get("geometry")}, self.device)
        self.kinds, self.sizes, self.friction = kinds, sizes, friction
        self.history, self.material = history.clone(), material
        self.x = history[:, -1].clone()
        self.v = (history[:, -1] - history[:, -2]) / self.dt
        self.active = wp.ones(len(history), dtype=int, device=self.device)
        if active_k == "auto":
            mu_val = float(self.material[3].item())
            stiff_val = float(self.material[1].item())
            frac = min(0.20, 0.045 * (1.0 + 1.5 * mu_val + 0.15 * stiff_val))
            self.active_k = max(256, min(len(history), int(round(len(history) * frac / 64.0)) * 64))
        elif active_k is None or int(active_k) >= len(history):
            self.active_k = None
        else:
            self.active_k = int(active_k)
        self.anchor_m = int(anchor_m)
        if self.active_k is not None:
            self.bg_fixed = torch.linspace(0, len(history) - 1, self.anchor_m, device=self.device).long()
        self.last_active_count = len(history) if self.active_k is None else self.active_k
        self.use_cuda_graph = use_cuda_graph and (self.active_k is None)
        self.graph = None
        self.last_displacement = None

    def _predict_displacement(self, inputs):
        """Replay the fixed-shape NN; feature values may change on every step."""
        if self.active_k is not None:
            nodes, pairs, inf = inputs
            support = float(self.config["features"]["support"])
            normal = pairs[:, 0, 1:4]
            rel_vel = pairs[:, 0, 4:7]
            approach = (normal * rel_vel).sum(dim=-1).clamp_min(0.0)
            pose_0 = self._current_poses[0]
            q_0 = pose_0[3:7]
            q_inv = torch.cat((-q_0[:3], q_0[3:4]), dim=-1)
            rel_x = self.x - pose_0[:3]
            cross_1 = 2.0 * torch.linalg.cross(q_inv[:3].unsqueeze(0).expand_as(rel_x), rel_x, dim=-1)
            local_x = rel_x + q_inv[3] * cross_1 + torch.linalg.cross(
                q_inv[:3].unsqueeze(0).expand_as(rel_x), cross_1, dim=-1
            )
            p_radius = 0.02 * float(self.material[11].item())
            q_box = local_x.abs() - self.sizes[0].unsqueeze(0)
            outside = q_box.clamp_min(0.0).norm(dim=-1)
            inside = q_box.amax(dim=-1).clamp_max(0.0)
            gap_box = outside + inside - p_radius
            gap_clamped = pairs[:, 0, 0] * support
            gap = torch.where(gap_clamped < support - 1e-4, gap_clamped, torch.maximum(gap_clamped, gap_box))

            mu_mat = float(self.material[3].item())
            stiff_norm = float(self.material[1].item())
            dilatancy = float(self.material[8].item())
            visc_norm = float(self.material[9].item())
            r_mat = (0.04 + 0.08 * mu_mat + 0.02 * stiff_norm + 0.03 * dilatancy) / (
                1.0 + 0.25 * visc_norm
            ) + 2.0 * self.dt * approach
            speed = (self.history[:, -1] - self.history[:, -2]).norm(dim=-1) / self.dt
            v_thresh = 0.015 * (1.0 + 0.5 * mu_mat)
            eval_score = (gap - r_mat) - 0.8 * (speed - v_thresh).clamp_min(0.0) - 2.0 * inf[:, 0]
            move_score = -5.0 * inf[:, 0] - 0.5 * (speed - v_thresh).clamp_min(0.0)
            active_candidates = torch.nonzero(eval_score < 0.0, as_tuple=False).squeeze(-1)
            if active_candidates.numel() > self.active_k:
                _, sub_idx = torch.topk(eval_score[active_candidates], k=self.active_k, largest=False, sorted=False)
                topk = active_candidates[sub_idx]
            elif active_candidates.numel() >= 128:
                topk = active_candidates
            else:
                _, topk = torch.topk(eval_score, k=min(128, self.active_k), largest=False, sorted=False)
            k_cur = int(topk.numel())
            d_k = gap[topk]
            r_k = r_mat[topk]
            tau = ((r_k - d_k) / 0.04).clamp(0.0, 1.0)
            gamma = tau * tau * (3.0 - 2.0 * tau)
            move_mask = torch.maximum((move_score[topk] < 0.0).float(), gamma)
            self.last_active_count = int((move_score[topk] < 0.0).sum().item())
            thresh = eval_score[topk].max()
            bg_scores = eval_score[self.bg_fixed]
            bg_valid = (bg_scores > thresh).float()
            if float(bg_valid.sum().item()) < 1.0:
                bg_valid = torch.ones_like(bg_valid)
            bg_mult = bg_valid * ((len(self.history) - k_cur) / bg_valid.sum().clamp_min(1.0))
            idx = torch.cat((topk, self.bg_fixed))
            mult = torch.cat((torch.ones(k_cur, device=self.device), bg_mult))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                disp_sub = self.model(nodes[idx], pairs[idx], inf[idx], multiplicity=mult).float()
                dx_raw = disp_sub[:k_cur]
                n_cup = torch.nn.functional.normalize(
                    -pairs[topk, 0, 1:4] + torch.tensor([0.0, 0.0, 0.35], device=self.device), dim=-1, eps=1e-6
                )
                u_into_cup = (-(dx_raw * n_cup).sum(dim=-1, keepdim=True)).clamp_min(0.0)
                dx_k = move_mask.unsqueeze(-1) * (dx_raw + (1.0 - tau.unsqueeze(-1)) * u_into_cup * n_cup)
            dx = torch.zeros_like(self.x)
            dx[self.bg_fixed] = disp_sub[k_cur:]
            dx[topk] = dx_k
            return dx.contiguous()

        if not self.use_cuda_graph:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(*inputs).contiguous()
        if self.graph is None:
            self._graph_inputs = tuple(value.clone() for value in inputs)
            current = torch.cuda.current_stream(self.x.device)
            capture_stream = torch.cuda.Stream(device=self.x.device)
            capture_stream.wait_stream(current)
            with torch.cuda.stream(capture_stream), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                for _ in range(3):
                    self.model(*self._graph_inputs)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=capture_stream):
                    self._graph_displacement = self.model(*self._graph_inputs).contiguous()
            current.wait_stream(capture_stream)
        for target, value in zip(self._graph_inputs, inputs, strict=True):
            if target.shape != value.shape:
                raise ValueError("Particle/collider counts changed; construct a new Predictor to recapture")
            target.copy_(value)
        self.graph.replay()
        return self._graph_displacement

    @torch.no_grad()
    def step(self, body_pose, body_velocity):
        """All colliders [B,7]/[B,6], or one box [7]/[6] with fixed-ground adapter."""
        if body_pose.ndim == 1:
            body_pose, body_velocity = box_ground_state(body_pose, body_velocity)
        self._current_poses = body_pose
        self._current_velocities = body_velocity
        inputs = features(
            self.history,
            body_pose,
            body_velocity,
            self.kinds,
            self.sizes,
            self.friction,
            self.material,
            self.dt,
            **self.config["features"],
            sdf=self.sdf,
        )
        displacement = self._predict_displacement(inputs)
        self.last_displacement = displacement
        torch_stream = torch.cuda.current_stream(self.x.device)
        warp_stream = wp.get_stream(str(self.x.device))
        if warp_stream.cuda_stream != torch_stream.cuda_stream:
            warp_stream = wp.stream_from_torch(torch_stream)
        with wp.ScopedStream(warp_stream):
            wp.launch(
                update,
                len(self.x),
                [
                    wp.from_torch(self.history),
                    wp.from_torch(self.x, dtype=wp.vec3),
                    wp.from_torch(self.v, dtype=wp.vec3),
                    wp.from_torch(displacement),
                    self.active,
                    self.dt,
                ],
                device=self.device,
            )
        return self.x, self.v

    def diagnostics(self):
        active_frac = float(self.last_active_count) / len(self.x)
        return dict(
            finite=bool(torch.isfinite(self.x).all()),
            active_fraction=active_frac,
            active_count=int(self.last_active_count),
        )
