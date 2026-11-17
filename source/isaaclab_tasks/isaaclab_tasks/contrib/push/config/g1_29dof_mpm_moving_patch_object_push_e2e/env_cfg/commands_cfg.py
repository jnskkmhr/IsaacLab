# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-frame velocity command; the arms have no command and hold a fixed pushing posture."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.push.config.g1_29dof_object_push_e2e import mdp as push_mdp

STANDING_COMMAND_THRESHOLD = 0.05
"""Planar command speed below which the robot is asked to stand [m/s]."""

HAND_CONTACT_FORCE_THRESHOLD = 1.0
"""Box-filtered force above which a wrist counts as touching the box [N]."""

BODY_CONTACT_FORCE_SCALE = 30.0
"""Non-wrist box contact force at which wrist-pushing credit has decayed to 1/e [N]."""

WRIST_LINKS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
"""Distal arm links that push the box; the no-hand asset has no bodies beyond them."""

NON_HAND_BODIES = [
    "pelvis.*",
    "torso_link",
    "waist_.*",
    "head_link",
    "logo_link",
    ".*_hip_.*",
    ".*_knee_link",
    ".*_ankle_.*",
    ".*_shoulder_.*",
    ".*_elbow_link",
]
"""Robot bodies that must not push the box; the three wrist links per arm are excluded."""


@configclass
class CommandsCfg:
    """One velocity command for the robot base; the box has no command of its own.

    The robot is asked to walk at this velocity, expressed in its own yaw frame. Because the box
    is in the way, tracking it means pushing the box at the same velocity, and the box-progress
    reward compares the box velocity with this same command.
    """

    base_velocity = push_mdp.PushVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=push_mdp.PushVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        # The fields below do not change the command. They select what the logged pushing
        # metrics measure: the box speed against this command, and wrist-versus-body contact.
        object_name="box",
        # The coupled-sand environment reports box forces per robot body; there is no contact sensor.
        contact_sensor_name="robot",
        hand_body_names=WRIST_LINKS,
        force_threshold=HAND_CONTACT_FORCE_THRESHOLD,
        command_threshold=STANDING_COMMAND_THRESHOLD,
    )
