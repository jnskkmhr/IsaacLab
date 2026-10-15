# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train all-particle position prediction with episode-separated validation.

Force labels are not used. The loss is next-position MSE divided by dt squared,
reported as velocity MSE for readable units. Near-body particles receive a
fixed additional weight; every particle contributes to training and inference.
"""

import argparse
import hashlib
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from geometry import body_contacts
from learning_data import EpisodeStore, velocity_history
from particle_model import ParticleGraph, graph_inputs


def one_step_metrics(model, store, device, stride=30):
    squared, persistence, gravity_error = [], [], []
    moving_squared = []
    model.eval()
    with torch.no_grad():
        for index in range(len(store)):
            e = store[index]
            for t in range(0, len(e["particle_position"]) - 1, stride):
                inputs = graph_inputs(
                    e["particle_position"][t],
                    velocity_history(e, t, model.config["history_length"]),
                    e,
                    e["body_pose"][t + 1],
                    e["body_velocity"][t + 1],
                    device,
                    model.config["radius"],
                )
                _, predicted = model(*inputs)
                target = e["particle_velocity"][t + 1]
                error = ((predicted.cpu().numpy() - target) ** 2).sum(axis=-1)
                squared.append(error.mean())
                persistence.append(((e["particle_velocity"][t] - target) ** 2).sum(axis=-1).mean())
                gravity_error.append(
                    ((e["particle_velocity"][t] + e["dt"] * e["gravity"] - target) ** 2).sum(axis=-1).mean()
                )
                body = e["moving_body"]
                distance, _, _ = body_contacts(
                    e["particle_position"][t], e["body_pose"][t + 1, body], e["body_size"][body], e["body_shapes"][body]
                )
                if np.any(distance < 0.03):
                    moving_squared.append(error[distance < 0.03].mean())
    model.train()
    return dict(
        velocity_vector_rmse_m_s=float(np.sqrt(np.mean(squared))),
        position_vector_rmse_m=float(np.sqrt(np.mean(squared))) * store[0]["dt"],
        near_body_velocity_vector_rmse_m_s=float(np.sqrt(np.mean(moving_squared))) if moving_squared else None,
        persistence_velocity_vector_rmse_m_s=float(np.sqrt(np.mean(persistence))),
        gravity_velocity_vector_rmse_m_s=float(np.sqrt(np.mean(gravity_error))),
        transitions=len(squared),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--unroll", type=int, default=1, help="Backpropagation through this many predicted particle steps."
    )
    parser.add_argument("--checkpoint-activations", action="store_true")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--input-velocity-scale", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.035)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument(
        "--early-transition-probability",
        type=float,
        default=0.0,
        help="Probability of sampling a rollout start from the first third of an episode; otherwise sample uniformly.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--evaluate-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--deadline", help="Optional timezone-aware ISO stop time; saves the final model.")
    args = parser.parse_args()
    if min(args.steps, args.unroll, args.hidden, args.passes, args.history_length, args.evaluate_every) < 1:
        parser.error("Training dimensions and step counts must be positive.")
    if min(args.radius, args.input_velocity_scale, args.learning_rate) <= 0 or args.noise < 0:
        parser.error("Scales/rate must be positive and noise nonnegative.")
    if not 0 <= args.early_transition_probability <= 1:
        parser.error("Early-transition probability must be between zero and one.")
    if (args.output / "metrics.jsonl").exists():
        parser.error("Use a new output directory; existing training logs will not be overwritten.")
    args.output.mkdir(parents=True, exist_ok=True)
    requested_stop = False

    def request_stop(signum, frame):
        nonlocal requested_stop
        requested_stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("train_particles.py", "particle_model.py", "geometry.py", "learning_data.py")
    }

    def save_checkpoint(checkpoint, filename):
        checkpoint["source_sha256"] = source_hashes
        path = args.output / filename
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParticleGraph(
        args.hidden,
        args.passes,
        args.radius,
        history_length=args.history_length,
        input_velocity_scale=args.input_velocity_scale,
    ).to(device)
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model = ParticleGraph(**saved["config"]).to(device)
        model.load_state_dict(saved["model"])
    model.config["activation_checkpointing"] = args.checkpoint_activations
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    train = EpisodeStore(args.data, "train", cache_size=256)
    validation = EpisodeStore(args.data, "validation", cache_size=16)
    split_names = {
        split: [f"{p.name}/{name}" for p, name in store.entries]
        for split, store in (("train", train), ("validation", validation))
    }
    (args.output / "inputs.json").write_text(
        json.dumps(dict(arguments=vars(args), episodes=split_names), default=str, indent=2)
    )
    best, losses = float("inf"), []
    start = time.monotonic()
    deadline = datetime.fromisoformat(args.deadline) if args.deadline else None
    if deadline is not None and deadline.tzinfo is None:
        parser.error("Deadline must include a UTC offset.")
    completed_step = 0
    log = (args.output / "metrics.jsonl").open("a", buffering=1)
    for step in range(1, args.steps + 1):
        if requested_stop or (deadline is not None and datetime.now(timezone.utc) >= deadline):
            break
        e = train[int(rng.integers(len(train)))]
        horizon = min(args.unroll, len(e["particle_position"]) - 1)
        available_starts = len(e["particle_position"]) - horizon
        if args.early_transition_probability and rng.random() < args.early_transition_probability:
            available_starts = max(1, min(available_starts, (len(e["particle_position"]) - 1) // 3))
        t = int(rng.integers(available_starts))
        x = e["particle_position"][t].copy()
        history = velocity_history(e, t, model.config["history_length"])
        if args.noise:
            history += np.tile(rng.normal(0, args.noise, (len(x), 3)), (1, model.config["history_length"]))
            x += rng.normal(0, args.noise * e["dt"], x.shape)
        x = torch.as_tensor(x, device=device)
        history = torch.as_tensor(history, device=device)
        loss = torch.zeros((), device=device)
        for offset in range(horizon):
            step_t = t + offset
            inputs = graph_inputs(
                x,
                history,
                e,
                e["body_pose"][step_t + 1],
                e["body_velocity"][step_t + 1],
                device,
                model.config["radius"],
            )
            predicted_position, predicted = model(*inputs)
            target_position = torch.as_tensor(e["particle_position"][step_t + 1], device=device)
            body = e["moving_body"]
            weight = 1 + 4 * (1 - inputs[3][:, body, 0].detach().clamp_min(0) / 0.3).clamp_min(0)
            # Position error is scaled by dt and horizon to compare losses
            # across rollout lengths. Gradients pass through prior predicted
            # particle states and SDF features; graph connectivity is discrete.
            error = (
                (predicted_position - target_position) / (e["dt"] * model.config["velocity_scale"] * (offset + 1))
            ) ** 2
            loss = loss + (error.mean(dim=-1) * weight).sum() / weight.sum() / horizon
            x = predicted_position
            history = torch.cat((history[:, 3:], predicted), dim=-1)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_step = step
        losses.append(float(loss.detach()))
        if step == 1 or step % 50 == 0:
            print(
                json.dumps(dict(step=step, loss=float(np.mean(losses[-50:])), seconds=time.monotonic() - start)),
                flush=True,
            )
        if step % args.evaluate_every == 0 or step == args.steps:
            metrics = one_step_metrics(model, validation, device)
            row = dict(
                step=step,
                elapsed_seconds=time.monotonic() - start,
                training_loss=float(np.mean(losses[-100:])),
                peak_gpu_tensor_memory_GB=torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
                validation=metrics,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            checkpoint = dict(
                config=model.config,
                model=model.state_dict(),
                step=step,
                validation=metrics,
                arguments=vars(args),
                episodes=split_names,
            )
            save_checkpoint(checkpoint, "last.pt")
            save_checkpoint(checkpoint, f"step_{step:06d}.pt")
            if metrics["velocity_vector_rmse_m_s"] < best:
                best = metrics["velocity_vector_rmse_m_s"]
                save_checkpoint(checkpoint, "best.pt")
    save_checkpoint(
        dict(
            config=model.config,
            model=model.state_dict(),
            step=completed_step,
            episodes=split_names,
            arguments=vars(args),
        ),
        "final.pt",
    )
    log.close()


if __name__ == "__main__":
    main()
