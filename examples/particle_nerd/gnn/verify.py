# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused GPU checks: complete neighborhoods, remote-body isolation and integration."""

import argparse
import json
from pathlib import Path

import torch
import warp as wp

from particle_nerd.gnn.features import Features
from particle_nerd.gnn.inference import advance
from particle_nerd.gnn.model import ParticleGNN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(41)
    x = torch.tensor([[0.0, 0.0, 0.1], [0.04, 0.0, 0.1], [0.2, 0.0, 0.1]], device="cuda")
    history = torch.zeros((3, 9), device="cuda")
    pose = torch.tensor([2.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], device="cuda")
    bv = torch.zeros(6, device="cuda")
    material = torch.ones(13, device="cuda")
    features = Features(3, capacity_per_particle=4)
    model = ParticleGNN(hidden=16, passes=2).cuda().eval()
    # Nonzero decoder makes the remote-body isolation test sensitive to features.
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    inputs = features.prepare(x, history, pose, bv, material, 5.0, compact=True)
    assert set(zip(inputs[4].tolist(), inputs[5].tolist())) == {(0, 1), (1, 0)}
    assert torch.all(inputs[3][:, 1] == 0)
    with torch.no_grad():
        reference = model(*inputs).clone()
        pose[0] = 3.0
        bv[:3] = 9.0
        distant = model(*features.prepare(x, history, pose, bv, material, 5.0, compact=True)).clone()
        torch.testing.assert_close(reference, distant, atol=0, rtol=0)
        padded = model(*features.prepare(x, history, pose, bv, material, 5.0)).clone()
        torch.testing.assert_close(reference, padded, atol=2e-6, rtol=2e-5)
    v = torch.ones_like(x) * 0.1
    history = v.repeat(1, 3)
    before = x.clone()
    correction = torch.zeros((3, 6), device="cuda")
    correction[:, :3] = 0.02
    correction[:, 3:] = -0.01
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        wp.launch(
            advance,
            3,
            [
                wp.from_torch(x, dtype=wp.vec3),
                wp.from_torch(v, dtype=wp.vec3),
                wp.from_torch(history),
                wp.from_torch(correction),
                0.02,
                features.overflow,
            ],
        )
    torch.testing.assert_close(x, before + 0.02 * 0.12)
    torch.testing.assert_close(v, torch.full_like(v, 0.09))
    torch.testing.assert_close(history[:, -3:], v)
    assert torch.all(history[:, :6] == 0.1)
    out = dict(
        neighborhood_matches_known_radius_graph=True,
        distant_body_has_zero_direct_influence=True,
        compact_and_padded_graph_predictions_match=True,
        warp_position_velocity_history_update_correct=True,
        device=torch.cuda.get_device_name(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
