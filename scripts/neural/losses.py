# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training losses of the Particle NeRD network, and the long resting rollout.

A particle counts as moving when the reference moves it more than 0.5 mm over the window. Moving and stationary
particles get equal weight, so the few particles near the object are not outvoted by the resting bed.
"""

import torch
from rollout import rollout


def balanced_loss(predicted, target, initial, dt):
    """Squared position error per step, in velocity units, averaged over the moving and the stationary particles.

    Args:
        predicted: Predicted positions [T, N, 3].
        target: Reference positions [T, N, 3].
        initial: Positions [N, 3] at the start of the window.
        dt: Time step [s].
    """
    moving = torch.linalg.vector_norm(target - initial[None], dim=-1) > 0.0005
    error = ((predicted - target) / dt).square().sum(-1)
    counts = torch.stack((moving.sum(), (~moving).sum()))
    sums = torch.stack(((error * moving).sum(), (error * ~moving).sum()))
    return (sums / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)


def world_particle_loss(predicted, predicted_vel, target, initial, dt):
    """Training loss of one window: ``0.6 position + 0.25 velocity + 0.15 rest drift``.

    The position and velocity terms are balanced between moving and stationary particles as in
    :func:`balanced_loss`. The rest-drift term penalizes any step-to-step motion of the particles that are at
    rest in the reference (slower than 1 mm/s).
    """
    loss_pos = balanced_loss(predicted, target, initial, dt)
    targ_seq = torch.cat((initial[None], target), dim=0)
    vel_targ = (targ_seq[1:] - targ_seq[:-1]) / dt
    vel_err = (predicted_vel - vel_targ).square().sum(-1)
    moving = torch.linalg.vector_norm(target - initial[None], dim=-1) > 0.0005
    counts = torch.stack((moving.sum(), (~moving).sum()))
    vel_sums = torch.stack(((vel_err * moving).sum(), (vel_err * ~moving).sum()))
    loss_vel = (vel_sums / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)

    pred_seq = torch.cat((initial[None], predicted), dim=0)
    fd_vel_pred = (pred_seq[1:] - pred_seq[:-1]) / dt
    at_rest = torch.linalg.vector_norm(vel_targ, dim=-1) < 0.001
    rest_drift = ((fd_vel_pred.square().sum(-1)) * at_rest).sum() / at_rest.sum().clamp_min(1)
    return 0.6 * loss_pos + 0.25 * loss_vel + 0.15 * rest_drift


def rest_unroll(model, dataset, rest_indices, rng, device, steps, segment):
    """Roll a resting bed out over ``steps`` steps and accumulate the gradients of the loss in place.

    The rollout is cut into segments of ``segment`` steps. The state that the model predicted is carried from one
    segment to the next (detached), so later segments start from the drift the model itself produced. Every
    segment is compared with the reference and its gradient is accumulated before its graph is freed, so the
    memory is that of one short window. The caller must not call ``backward`` on the returned loss.

    Args:
        model: The network.
        dataset: Training windows.
        rest_indices: Indices of the episodes to draw from.
        rng: Random generator that picks the episode and its start.
        device: Device of the window.
        steps: Length of the rollout.
        segment: Length of one gradient segment.

    Returns:
        The loss, averaged over the segments, without a graph.
    """
    idx = int(rng.choice(rest_indices))
    length = dataset.num_frames(idx)
    record = dataset.records[idx][0]
    earliest = round(float(record.get("rest_start_s", 0.0)) / float(record["dt_s"]))
    start = int(rng.integers(min(earliest, length - steps - 1), length - steps))
    sample = dataset.sequence(rng, steps, device, index=idx, start=start)
    history, dt, count = sample["history"], sample["dt"], steps // segment
    total = 0.0
    for k in range(count):
        a, b = k * segment, (k + 1) * segment
        part = dict(sample)
        part["history"] = history
        part["poses"], part["velocities"], part["targets"] = (
            sample["poses"][a:b],
            sample["velocities"][a:b],
            sample["targets"][a:b],
        )
        reference_start = sample["history"][:, -1] if k == 0 else sample["targets"][a - 1]
        pred, pred_vel, history = rollout(model, part, return_vel=True, return_history=True)
        loss = world_particle_loss(pred, pred_vel, part["targets"], reference_start, dt) / count
        loss.backward()
        total += float(loss.detach())
        history = history.detach()
    return torch.tensor(total, device=device)
