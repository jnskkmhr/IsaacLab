# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep the MPM critic's one-dimensional material placeholder."""

import torch

from isaaclab.envs import ManagerBasedRLEnv


def terrain_material_parameters(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return zeros, matching the reference MPM critic rather than exposing RFT stiffness."""
    return torch.zeros(env.num_envs, 1, device=env.device)
