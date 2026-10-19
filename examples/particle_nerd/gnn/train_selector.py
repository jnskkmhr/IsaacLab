# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a particle activity classifier on current-state features, with future motion labels."""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from particle_nerd.gnn.data import Episodes
from particle_nerd.gnn.features import Features
from particle_nerd.gnn.region_selection import MotionSelector, reference_activity, selection_inputs


@torch.no_grad()
def validate(model, episodes, features, horizon, distance):
    model.eval()
    rng = np.random.default_rng(6301)
    thresholds = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    counts = np.zeros((len(thresholds), 4), dtype=np.int64)
    total_loss = 0.0
    for index, record in enumerate(episodes.records):
        sample = episodes.sample(rng, episode=index)
        with h5py.File(record["path"]) as f:
            target = torch.as_tensor(reference_activity(f, sample["frame"], horizon, distance), device="cuda")
        inputs = features.prepare(
            sample["x"],
            sample["history"],
            sample["pose"],
            sample["body_velocity"],
            sample["material"],
            sample["mass"],
            compact=True,
        )
        logits = model(selection_inputs(inputs))
        total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=torch.tensor(5.0, device="cuda")
        ).item()
        probabilities = logits.sigmoid().cpu().numpy()
        truth = target.cpu().numpy()
        for j, t in enumerate(thresholds):
            predicted = probabilities >= t
            counts[j] += [
                np.sum(predicted & truth),
                np.sum(predicted & ~truth),
                np.sum(~predicted & truth),
                np.sum(~predicted & ~truth),
            ]
    rows = []
    for t, (tp, fp, fn, tn) in zip(thresholds, counts):
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        rows.append(
            dict(
                threshold=float(t),
                precision=precision,
                recall=recall,
                false_positive_rate=float(fp / max(1, fp + tn)),
                active_fraction=float((tp + fp) / max(1, tp + fp + fn + tn)),
                f2=5 * precision * recall / max(1e-12, 4 * precision + recall),
            )
        )
    eligible = [r for r in rows if r["recall"] >= 0.95]
    chosen = (
        max(eligible, key=lambda r: (r["precision"], r["threshold"])) if eligible else max(rows, key=lambda r: r["f2"])
    )
    model.train()
    return dict(
        loss=total_loss / len(episodes.records),
        transitions=len(episodes.records),
        thresholds=rows,
        selected=chosen,
        selection_rule="Highest precision with at least 95% validation recall; maximum F2 if none qualify",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--evaluate-every", type=int, default=500)
    p.add_argument("--label-horizon", type=int, default=5)
    p.add_argument("--label-distance", type=float, default=0.0005)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(8537)
    rng = np.random.default_rng(8537)
    train = Episodes(args.data, "train")
    validation = Episodes(args.data, "validation")
    assert not ({r["index"] for r in train.records} & {r["index"] for r in validation.records})
    features = Features(16384)
    model = MotionSelector().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best = float("inf")
    started = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        sample = train.sample(rng)
        record = next(r for r in train.records if r["index"] == sample["index"])
        with h5py.File(record["path"]) as f:
            labels = reference_activity(f, sample["frame"], args.label_horizon, args.label_distance)
        target = torch.as_tensor(labels, device="cuda", dtype=torch.float32)
        inputs = features.prepare(
            sample["x"],
            sample["history"],
            sample["pose"],
            sample["body_velocity"],
            sample["material"],
            sample["mass"],
            compact=True,
        )
        logits = model(selection_inputs(inputs))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=torch.tensor(5.0, device="cuda")
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(
                json.dumps(
                    dict(
                        event="train_selector",
                        step=step,
                        loss=float(np.mean(losses[-100:])),
                        elapsed_s=time.time() - started,
                    )
                ),
                flush=True,
            )
        if step % args.evaluate_every == 0 or step == args.steps:
            metrics = validate(model, validation, features, args.label_horizon, args.label_distance)
            row = dict(
                step=step,
                elapsed_s=time.time() - started,
                train_loss=float(np.mean(losses[-args.evaluate_every :])),
                **metrics,
            )
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            saved = dict(
                model=model.state_dict(),
                config=model.config,
                step=step,
                threshold=metrics["selected"]["threshold"],
                label_horizon=args.label_horizon,
                label_distance=args.label_distance,
                validation=metrics,
                train_episodes=len(train.records),
                validation_episodes=len(validation.records),
                input=(
                    "Current particle velocity history, rigid geometry/motion features and "
                    "neighboring particle speeds; no future reference state"
                ),
                output="Per-particle activity probability for the next 0.1 seconds",
            )
            torch.save(saved, args.output / "last.pt")
            if metrics["loss"] < best:
                best = metrics["loss"]
                torch.save(saved, args.output / "best.pt")
            print(
                json.dumps(
                    dict(
                        event="selector_validation",
                        step=step,
                        loss=metrics["loss"],
                        **metrics["selected"],
                    )
                ),
                flush=True,
            )
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                steps=args.steps,
                elapsed_s=time.time() - started,
                best_validation_loss=best,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
