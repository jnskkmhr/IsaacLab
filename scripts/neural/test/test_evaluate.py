# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the numbers that the evaluation reports."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
import metrics  # noqa: E402

DT = 0.005


def drifting_rollout(steps: int) -> tuple[np.ndarray, np.ndarray]:
    """A rollout whose error grows by 0.01 mm per step."""
    reference = np.zeros((steps + 1, 5, 3), dtype=np.float32)
    predicted = reference.copy()
    predicted[:, :, 0] = np.arange(steps + 1, dtype=np.float32)[:, None] * 1e-5
    return predicted, reference


@pytest.mark.parametrize("steps, after_1s, at_end", [(200, 2.0, 2.0), (400, 2.0, 4.0), (60, 0.6, 0.6)])
def test_rollout_of_any_length_is_reported_and_summarized(steps, after_1s, at_end):
    """A rollout that ends at exactly 1 s, or before it, still has every number that the summary averages."""
    errors = metrics.position_errors(*drifting_rollout(steps), DT)
    row = {"category": "rest", "terrain": "flat", "failed": False, "finite": True, **errors}
    row.update(top3_dz_mm=-1.0, top3_dz_mpm_mm=-0.5, mid_dz_mm=-0.2, floor_dz_mm=0.0)

    summary = evaluate.summarize([row])

    assert errors["rmse_mm@1s"] == pytest.approx(after_1s, rel=1e-4)
    assert errors["rmse_mm@end"] == pytest.approx(at_end, rel=1e-4)
    assert summary["rest/flat"]["rmse_mm@1s"] == pytest.approx(after_1s, abs=0.005)
    assert summary["rest/all"]["files"] == 1


def test_layer_drop_separates_the_layers_of_the_bed():
    """Sinking of the surface must not be hidden by the layers below it."""
    depth = np.linspace(-0.30, 0.0, 61, dtype=np.float32)
    start = np.stack((np.zeros_like(depth), np.zeros_like(depth), depth), axis=-1)
    end = start.copy()
    end[depth > -0.031, 2] -= 0.004  # only the top 3 cm sink, by 4 mm
    bed = np.ones(len(start), dtype=bool)

    drop = metrics.layer_drop(np.stack((start, end)), start, bed, 1)

    assert drop["top3"] == pytest.approx(-4.0, abs=1e-3)
    assert drop["mid"] == pytest.approx(0.0, abs=1e-6)
    assert drop["floor"] == pytest.approx(0.0, abs=1e-6)
