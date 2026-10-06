# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check environment selection when resetting Warp soft-contact history."""

import pytest
import torch
import warp as wp

from isaaclab_tasks.contrib.soft_contact._impl import soft_contact_model_warp as models
from isaaclab_tasks.contrib.soft_contact._impl.collider import PlaneColliderCfg
from isaaclab_tasks.contrib.soft_contact._impl.material import DefaultConeDRFTCfg, Material3DRFTCfg, PoppySeedCPCfg


@pytest.mark.parametrize(
    "solver_type,material_type,history_fields",
    [
        (models.RFT_3D, Material3DRFTCfg, ("alpha_unfiltered", "alpha_filtered")),
        (models.RFT_2D, PoppySeedCPCfg, ("force_gm", "force_ema")),
        (models.ConeDRFT, DefaultConeDRFTCfg, ("z_max", "force_gm", "force_ema")),
        (models.ConeDRFTMultiPoint, DefaultConeDRFTCfg, ("z_max", "force_gm", "force_ema")),
    ],
)
def test_reset_preserves_unselected_environments(solver_type, material_type, history_fields):
    """Slices, optional IDs, and integer tensors reset only the selected contact histories."""
    with wp.ScopedDevice("cpu"):
        collider = PlaneColliderCfg(
            contact_edge_x=(-0.1, 0.1), contact_edge_y=(-0.05, 0.05), contact_edge_z=(-0.02, 0.0), resolution=(2, 2)
        )
        solver = solver_type(4, 2, "cpu", 0.01, material_type(), collider)
    histories = [
        wp.to_torch(getattr(solver, name)) for name in (*history_fields, "tau_r", "contact_point_lin_vel_prev")
    ]
    for env_ids, reset_mask in [
        (slice(None), [True, True, True, True]),
        (slice(1, None, 2), [False, True, False, True]),
        (torch.tensor([0, 2], dtype=torch.int32), [True, False, True, False]),
        (slice(2, 2), [False, False, False, False]),
        (None, [True, True, True, True]),
    ]:
        for history in histories:
            history.fill_(1.0)
        solver.reset(env_ids)
        for history in histories:
            expected = torch.ones_like(history)
            expected[torch.tensor(reset_mask)] = 0.0
            torch.testing.assert_close(history, expected)
