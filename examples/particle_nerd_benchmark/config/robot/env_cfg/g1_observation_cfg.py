# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM policy observation contract: 10-frame term histories and a seven-value command."""

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .g1_action_cfg import ACTIVE_JOINT


def command(env):
    result = torch.zeros((env.num_envs, 7), device=env.device)
    result[:, 0] = env.cfg.forward_speed
    result[:, 6] = 1.0  # Fixed world heading: identity quaternion in xyzw order.
    return result


@configclass
class G1ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        )
        actions = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False
        history_length = 10
        history_order = "term"

    @configclass
    class CommandCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(func=command)
        enable_corruption = False

    policy = PolicyCfg()
    command = CommandCfg()
