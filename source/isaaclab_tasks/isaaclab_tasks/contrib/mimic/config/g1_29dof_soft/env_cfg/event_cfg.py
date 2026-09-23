# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp
from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.event_cfg import G1EventCfg as RigidEventCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft import mdp as soft_mdp

from ..mdp.events import perturb_ankle_pitch_in_stance, push_body_in_stance, push_by_setting_velocity_in_stance


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

    # finetuning
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

@configclass
class G1EventFinetuneCfg(G1EventCfg):
    """Preserve motion resets and randomize the soft material each episode."""

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
    push_foot = EventTerm(
        func=push_body_in_stance,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "force_range": {"x": (-100.0, 100.0), "y": (-100.0, 100.0), "z": (-0.0, 0.0)},
            "torque_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_.*"]),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_ankle_roll_.*"],
                preserve_order=True,
            ),
        },
    )

    perturb_ankle_pitch = EventTerm(
        func=perturb_ankle_pitch_in_stance,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "torque_range": (-5.0, 5.0),
            "duration_range_s": (0.2, 0.5),
        },
    )
