# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton reference task with current-only commands and left/right augmentation."""

from isaaclab.utils import configclass

from .env_cfg.action_cfg import G1SymmetricActionsCfg
from .env_cfg.symmetry_observation_cfg import G1SymmetricObservationsCfg
from .flat_env_cfg import G1FlatEnvCfg


@configclass
class G1FlatSymmetryEnvCfg(G1FlatEnvCfg):
    actions = G1SymmetricActionsCfg()
    observations = G1SymmetricObservationsCfg()
