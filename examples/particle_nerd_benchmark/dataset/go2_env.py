# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 on MPM sand with the Isaac Lab velocity-task policy contract (48 observations, joint-position actions)."""

import torch
from isaaclab_newton.sim.schemas import NewtonCollisionCfg

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from ..config.robot.env_cfg.event_cfg import RobotEventsCfg
from ..config.robot.quadruped_env_cfg import QuadrupedEnvCfg


def velocity_command(env) -> torch.Tensor:
    """Constant base-velocity command (vx, vy, yaw rate) of the current episode."""
    return env.cfg.go2_command.to(env.device).expand(env.num_envs, -1)


@configclass
class Go2ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class Go2ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        velocity_commands = ObservationTermCfg(func=velocity_command)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2PolicyEnvCfg(QuadrupedEnvCfg):
    """Benchmark quadruped scene and sand physics with the trained policy's actions and observations."""

    actions: Go2ActionsCfg = Go2ActionsCfg()
    observations: Go2ObservationsCfg = Go2ObservationsCfg()
    events: RobotEventsCfg = RobotEventsCfg()
    go2_command: torch.Tensor = torch.zeros((1, 3))

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        # The Newton actuator path folds all substeps into one physics call; the Python path steps them one by one,
        # which lets the collector record every 200 Hz state.
        self.sim.use_newton_actuators = False
        robot = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        robot.spawn.activate_contact_sensors = False
        robot.spawn.collision_props = {r"/.*foot/.*": [NewtonCollisionCfg(contact_margin=0.02, contact_gap=0.0)]}
        robot.init_state.pos = (-1.0, 0.0, 0.36)
        self.scene.robot = robot
