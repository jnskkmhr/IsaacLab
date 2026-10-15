# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect peak-error meaning and the declared force/motion selection rule."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight_study import choose_models, impact_summary


def test_peak_error_uses_the_reference_peak_time():
    result = impact_summary(
        {
            "episode": {
                "estimated_force_at_reference_peak_N": 7.0,
                "reference_peak_upward_force_N": 10.0,
                "estimated_peak_upward_force_N": 12.0,
                "estimated_peak_time_error_s": 0.005,
            }
        }
    )
    assert result["peak_at_reference_time_rmse_N"] == 3.0
    assert result["peak_at_reference_time_bias_N"] == -3.0
    assert result["maximum_force_error_rmse_N"] == 2.0
    assert result["peak_timing_rmse_s"] == 0.005


def test_selection_retains_peak_tradeoff_and_rejects_worse_box_motion():
    def candidate(force, peak, position, orientation):
        return {
            "validation": {
                "metrics": {"box_force_vector_rmse_N": force, "solver_failures": 0},
                "impact": {"peak_at_reference_time_rmse_N": peak},
                "replay": {"valid": True, "position_rmse_m": position, "orientation_rmse_deg": orientation},
            }
        }

    state = {
        "candidates": {
            "baseline": candidate(10.0, 40.0, 0.01, 5.0),
            "mean_candidate": candidate(7.0, 60.0, 0.007, 3.0),
            "peak_candidate": candidate(8.0, 30.0, 0.008, 4.0),
            "bad_motion": candidate(5.0, 20.0, 0.02, 6.0),
        }
    }
    choose_models(state)
    assert state["best_mean"] == "mean_candidate"
    assert state["best_peak"] == "peak_candidate"
