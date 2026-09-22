# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math 

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.event_cfg import G1EventCfg as RigidEventCfg
import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft import mdp as soft_mdp

from ..mdp.events import push_by_setting_velocity_in_stance


@configclass
class G1EventCfg(RigidEventCfg):
    """Preserve motion resets and randomize the soft material each episode."""

    randomize_friction = EventTerm(
        func=soft_mdp.randomize_terrain_friction,
        mode="reset",
        params={"friction_range": (0.1, 1.0), "contact_solver_name": "physics_callback"},
    )
    randomize_stiffness = EventTerm(
        func=soft_mdp.randomize_terrain_stiffness,
        mode="reset",
        params={"stiffness_range": (0.2, 0.9), "contact_solver_name": "physics_callback"},
    )
    randomize_material_density = EventTerm(
        func=soft_mdp.randomize_material_density,
        mode="reset",
        params={
            "packing_ratio_range": (0.5, 1.0),
            "bulk_density_range": (1000.0, 3000.0),
            "contact_solver_name": "physics_callback",
        },
    )

    # finetune
    reset_base = EventTerm(
        func=mimic_mdp.reset_root_state_from_reference,
        mode="reset",
        params={
            "command_name": "motion",
            "pose_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )
    push_robot = EventTerm(
        func=push_by_setting_velocity_in_stance,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
        },
    )
