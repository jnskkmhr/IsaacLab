# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The RFT tow task on the rough terrain of its walking task, with a share of environments that tow nothing.

The walking task's rough-terrain configuration (rough surface with rigid and analytical-sand columns, terrain
curriculum, sand material randomization for the feet, surface-relative observations and rewards) gets the tow
task's parts: the sled with its own analytical sand force, the tether, the forward-only command, and the
orientation penalty that leaves a forward lean free. The sled's mass is drawn anew at every reset, and in a fixed
share of the environments the tether is cut, so walking without a load stays part of training.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.push.config.g1_29dof_mpm_tow.mdp as tow_mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact.rough_env_cfg import G1RoughEnvCfg

from .env_cfg import G1ActionsCfg, TetherCfg
from .env_cfg.reward_cfg import FORWARD_LEAN_LIMIT
from .env_cfg.scene_cfg import GROUND_FRICTION, SLED_ASSET, SLED_MASS, SLED_RESET_DISTANCE, G1TowSceneCfg

SLED_MASS_RANGE = (1.0, SLED_MASS)
"""Lowest and highest sled mass drawn at a reset [kg]."""
NO_LOAD_EVERY = 4
"""Every this many environments, one has the tether cut for the whole run: a quarter tows nothing."""
SUPPORT_DEPTH_RANGE = (0.0, 0.4)
"""Depth of the rigid support under the analytical sand at the first and last terrain row [m]; the value of the
walking policy the tow policy starts from."""


@configclass
class G1TowRoughEnvCfg(G1RoughEnvCfg):
    """G1 tows a sled of random mass over the walking task's rough rigid and analytical-sand terrain."""

    tether: TetherCfg = TetherCfg()
    """The rope between the robot and the sled."""
    camera_follow_eye: tuple[float, float, float] | None = (3.0, 3.0, 2.0)
    """Viewer position relative to the last robot's root [m]; None keeps a fixed view."""
    no_load_every: int = NO_LOAD_EVERY
    """Every this many environments, one has the tether cut."""

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.scene.sled = G1TowSceneCfg().sled
        self.scene.terrain.support_depth_range = SUPPORT_DEPTH_RANGE
        # Contact friction is the larger of the two shapes' values: the ground must not exceed the sled's.
        self.scene.terrain.physics_material = RigidBodyMaterialCfg(
            static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION
        )
        # The sled is a second intruder in the same analytical sand, measured against the rough surface. The
        # rope term comes after both contact terms: they replace the permanent wrenches, including the rope force.
        tow_actions = G1ActionsCfg()
        self.actions.sled_physics_callback = tow_actions.sled_physics_callback.replace(terrain_name="terrain")
        self.actions.tether_hold = tow_actions.tether_hold
        # Forward only: the sled is behind the robot and the rope cannot push.
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z = (0.0, 1.0), (0.0, 0.0), (-0.5, 0.5)
        # As in the flat tow tasks: no command or tracking-reward ramps, and a free forward lean.
        for name in ("command_vel", "track_lin_vel", "track_ang_vel", "track_heading", "track_lin_vel_weight"):
            setattr(self.curriculum, name, None)
        self.rewards.flat_orientation_l2 = RewTerm(
            func=tow_mdp.lean_orientation_l2, weight=-10.0, params={"forward_lean_limit": FORWARD_LEAN_LIMIT}
        )
        # After reset_base: the sled goes behind wherever the robot was put, on the terrain surface there.
        self.events.reset_sled = EventTerm(
            func=tow_mdp.reset_object_behind_robot,
            mode="reset",
            params={
                "distance": SLED_RESET_DISTANCE,
                "clearance": 0.02,
                "asset_cfg": SceneEntityCfg(SLED_ASSET),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        self.events.randomize_sled_mass = EventTerm(
            func=tow_mdp.randomize_sled_mass,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(SLED_ASSET), "mass_range": SLED_MASS_RANGE},
        )
