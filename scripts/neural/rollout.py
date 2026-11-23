# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recurrent rollouts of the Particle NeRD network on a window of :class:`dataset.EpisodeWindows`.

A rollout feeds the network its own predictions: the reference state is used for the start only. The history
carries the predicted velocity, by storing ``x_next - dt * v_next`` as the previous position.

- :func:`rollout` keeps the gradients and is used for training and validation.
- :func:`predict` is the evaluation rollout. It can put slow particles to sleep and can step only an active set
  of particles, as the Isaac Lab solver does.
"""

import time

import numpy as np
import torch


def rollout(model, sample, return_vel=False, return_history=False):
    """Roll the model out over the window ``sample`` without replacing its state by the reference.

    Returns:
        The predicted positions [T, N, 3]; with ``return_vel`` also the predicted velocities [T, N, 3]; with
        ``return_history`` also the history [N, 3, 3] after the last step, to continue from.
    """
    history = sample["history"]
    predicted = []
    predicted_vels = []
    for pose, velocity in zip(sample["poses"], sample["velocities"], strict=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            current, vel_next = model.step_from_history(
                history,
                pose.contiguous(),
                velocity.contiguous(),
                sample["kinds"],
                sample["sizes"],
                sample["friction"],
                sample["material"],
                dt=sample["dt"],
                return_vel=True,
            )
        predicted_vels.append(vel_next)
        predicted.append(current)
        prev_synth = current - sample["dt"] * vel_next
        history = torch.stack((history[:, 1], prev_synth, current), dim=1).contiguous()
    if return_history:
        return torch.stack(predicted), torch.stack(predicted_vels), history
    if return_vel:
        return torch.stack(predicted), torch.stack(predicted_vels)
    return torch.stack(predicted)


def select_active(history, pose, half_size, dt, k_act, anchors):
    """Active set of the Particle NeRD solver (``nerd_solver.py``), for one environment.

    The ``k_act`` particles with the lowest score (distance to the object box minus 0.12 times the speed) are
    active. ``anchors`` background particles, evenly spread over the particle index, are added as extra tokens;
    each carries a weight so that together they stand in for every particle outside the active set. Returns the
    active indices, the token indices (active then anchors), the token weights (None without anchors) and the
    blend factor of each active particle, which fades the update out towards the selection cut-off.
    """
    pos, prev = history[:, -1], history[:, -2]
    n = pos.shape[0]
    speed = torch.linalg.vector_norm(pos - prev, dim=-1) / dt
    axis, w = pose[3:6].expand_as(pos), pose[6]
    rel = pos - pose[:3]
    twice = 2.0 * torch.linalg.cross(axis, rel)
    local = rel - w * twice + torch.linalg.cross(axis, twice)  # inverse rotation by the quaternion (x, y, z, w)
    q = local.abs() - half_size
    gap = torch.linalg.vector_norm(q.clamp_min(0.0), dim=-1) + q.amax(dim=-1).clamp_max(0.0)
    score = gap - 0.12 * speed.clamp_max(2.0)
    top_vals, topk = torch.topk(score, k=k_act, largest=False, sorted=False)
    thresh = top_vals.amax()
    tau = ((thresh - top_vals) / 0.015).clamp(0.0, 1.0)
    gamma = tau * tau * (3.0 - 2.0 * tau)
    if anchors <= 0:
        return topk, topk, None, gamma
    background = torch.linspace(0, n - 1, anchors, device=pos.device).long()
    valid = (score[background] > thresh).float()
    if valid.sum() < 1.0:
        valid = torch.ones_like(valid)
    weights = torch.cat((torch.ones_like(top_vals), valid * (float(n - k_act) / valid.sum().clamp_min(1.0))))
    return topk, torch.cat((topk, background)), weights, gamma


def active_step(model, history, pose, velocity, sample, selection):
    """One step in which only the active particles move; the others keep their position and get zero velocity."""
    topk, index, weights, gamma = selection
    next_sub, vel_sub = model.step_from_history_batched(
        history[index][None].contiguous(),
        pose[None].contiguous(),
        velocity[None].contiguous(),
        sample["kinds"],
        sample["sizes"],
        sample["friction"],
        sample["material"],
        dt=sample["dt"],
        multiplicity=None if weights is None else weights[None].contiguous(),
        return_vel=True,
    )
    k = topk.shape[0]
    current = history[:, -1].clone()
    vel_next = torch.zeros_like(current)
    current[topk] += gamma[:, None] * (next_sub[0, :k].float() - current[topk])
    vel_next[topk] = gamma[:, None] * vel_sub[0, :k].float()
    return current, vel_next


@torch.no_grad()
def predict(model, sample, sleep_speed=0.0, active=None, step_seconds=None):
    """Roll the model out over the whole window; stop on a nonfinite or unbounded state.

    Args:
        model: The network.
        sample: Window of :meth:`dataset.EpisodeWindows.sequence`.
        sleep_speed: Particle sleeping [m/s]. A particle whose predicted move in one step is slower than this
            stays exactly where it is and gets zero velocity. It moves again as soon as a later prediction is
            faster. 0 disables sleeping.
        active: None to step every particle, or ``{"k": K, "anchors": M, "interval": I}`` to step the active set
            of the Isaac Lab solver: the top K particles plus M weighted background anchors, with the features
            computed on those particles only, reselected every I steps.
        step_seconds: If a list, the time of every model step after the first ten is appended to it. The device
            is synchronized around each step, and copying the result to the host is not included.

    Returns:
        The positions [T + 1, N, 3] including the start frame, and None or a description of the failure.
    """
    history = sample["history"]
    positions = [history[:, -1].cpu().numpy()]
    failure = None
    selection = None
    for step, (pose, velocity) in enumerate(zip(sample["poses"], sample["velocities"], strict=True)):
        if step_seconds is not None:
            torch.cuda.synchronize()
            tic = time.perf_counter()
        if active is not None:
            if step % max(1, int(active.get("interval", 4))) == 0:
                selection = select_active(
                    history, pose[0], sample["sizes"][0], sample["dt"], int(active["k"]), int(active["anchors"])
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                current, vel_next = active_step(model, history, pose, velocity, sample, selection)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                current, vel_next = model.step_from_history(
                    history,
                    pose.contiguous(),
                    velocity.contiguous(),
                    sample["kinds"],
                    sample["sizes"],
                    sample["friction"],
                    sample["material"],
                    dt=sample["dt"],
                    return_vel=True,
                )
        if sleep_speed > 0.0:
            previous = history[:, -1]
            asleep = torch.linalg.vector_norm(current.float() - previous, dim=-1, keepdim=True) < (
                sleep_speed * sample["dt"]
            )
            current = torch.where(asleep, previous, current.float())
            vel_next = torch.where(asleep, torch.zeros_like(vel_next), vel_next)
        if step_seconds is not None:
            torch.cuda.synchronize()
            if step >= 10:
                step_seconds.append(time.perf_counter() - tic)
        x = current.cpu().numpy()
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 5.0:
            failure = {
                "time_s": (step + 1) * sample["dt"],
                "reason": "Nonfinite particle state or a particle more than 5 m from the tray origin",
            }
            break
        positions.append(x)
        prev_synth = current - sample["dt"] * vel_next
        history = torch.stack((history[:, 1], prev_synth, current), dim=1).contiguous()
    return np.stack(positions), failure
