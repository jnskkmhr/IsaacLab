# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset presets for already-retargeted G1 exports, independent of human skeleton formats."""

from collections.abc import Callable
from dataclasses import MISSING

import numpy as np

from isaaclab.utils.configclass import configclass

from .util.importers import load_motion


@configclass
class DatasetTermCfg:
    """Importer callable receiving a path and returning canonical motion arrays."""

    func: Callable[..., dict[str, np.ndarray]] = MISSING
    """Import function for already-retargeted robot motions."""

    params: dict = {}
    """Keyword arguments passed to func, including source labels and export conventions."""


# Exporters may use different field names, quaternion orders, and joint orders for
# the same dataset. Configure these in params rather than inferring them from its name.
LAFAN_CFG = DatasetTermCfg(func=load_motion)
CMU_BVH_CFG = DatasetTermCfg(func=load_motion)
KIMODO_CFG = DatasetTermCfg(func=load_motion)

DATASET_CFGS = {"lafan": LAFAN_CFG, "cmu_bvh": CMU_BVH_CFG, "kimodo": KIMODO_CFG}
