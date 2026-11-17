# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ground-only mesh scans for homogeneous, replicated fixed MPM beds."""

import numpy as np
from isaaclab_newton.sensors import LegacyMultiMeshRayCaster

import isaaclab.sim as sim_utils


class FixedBedHeightScanner(LegacyMultiMeshRayCaster):
    """Place static prototype meshes at each world's origin; track trays normally.

    The legacy mesh scanner handles tracked ClonePlan targets, but static targets
    fall back to authored USD. With physics replication only env_0 is authored,
    so its static mesh pose otherwise gets reused in every world. These testbeds
    have identical geometry in every world and differ only by translation.
    """

    def _build_mesh_records(self, target_cfg, plan, dummy_mesh_id):
        records, dummy_mesh_id, tracked = super()._build_mesh_records(target_cfg, plan, dummy_mesh_id)
        if plan is not None and plan.positions is not None and not target_cfg.track_mesh_transforms:
            prims = sim_utils.find_matching_prims(target_cfg.prim_expr)
            if len(prims) == 1 and str(prims[0].GetPath()).startswith(plan.env_template.format(0) + "/"):
                for env_id, row in enumerate(records):
                    offset = np.asarray(plan.positions[env_id]) - np.asarray(plan.positions[0])
                    records[env_id] = [(mesh, tuple(np.asarray(pos) + offset), quat) for mesh, pos, quat in row]
        return records, dummy_mesh_id, tracked
