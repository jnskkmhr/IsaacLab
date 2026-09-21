# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.event_cfg import G1EventCfg as RigidEventCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft import mdp as soft_mdp


@configclass
class G1EventCfg(RigidEventCfg):
    """Preserve motion resets and randomize the soft material each episode."""

    randomize_friction = EventTermCfg(
        func=soft_mdp.randomize_terrain_friction,
        mode="reset",
        params={"friction_range": (0.1, 1.0), "contact_solver_name": "physics_callback"},
    )
    randomize_stiffness = EventTermCfg(
        func=soft_mdp.randomize_terrain_stiffness,
        mode="reset",
        params={"stiffness_range": (0.2, 0.9), "contact_solver_name": "physics_callback"},
    )
    randomize_material_density = EventTermCfg(
        func=soft_mdp.randomize_material_density,
        mode="reset",
        params={
            "packing_ratio_range": (0.5, 1.0),
            "bulk_density_range": (1000.0, 3000.0),
            "contact_solver_name": "physics_callback",
        },
    )
