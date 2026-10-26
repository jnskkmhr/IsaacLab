# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Adapted from Agile. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg, G1RoughEnvCfg_PLAY


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    """Legacy flat task entry point using the exact Agile reference terrain and dynamics."""


@configclass
class G1FlatEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    """Play the reference task through the legacy flat entry point."""
