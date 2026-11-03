# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat moving-patch whole-body pushing task."""

import isaaclab.envs.mdp as core_mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.commands_cfg import G1CommandsCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.event_cfg import G1EventCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.observation_cfg import G1ObservationsCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.physics_cfg import G1PhysicsProxyCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.reward_cfg import G1RewardsCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.termination_cfg import G1TerminationsCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mpm_env_cfg import G1MovingPatchEnvCfg

from . import mdp
from .commands import BoxHandPoseCommandCfg
from .scene_cfg import G1ObjectPushSceneCfg


@configclass
class CommandsCfg(G1CommandsCfg):
    hand_pose = BoxHandPoseCommandCfg()


@configclass
class ManipulationCfg(ObservationGroupCfg):
    object_pose = ObservationTermCfg(func=mdp.object_pose_b)
    hand_targets = ObservationTermCfg(func=core_mdp.generated_commands, params={"command_name": "hand_pose"})
    hand_pose = ObservationTermCfg(func=mdp.hand_pose_b)
    concatenate_terms = True
    enable_corruption = False


@configclass
class ObjectPrivilegedCfg(ObservationGroupCfg):
    velocity = ObservationTermCfg(func=mdp.box_velocity_b)
    concatenate_terms = True
    enable_corruption = False


@configclass
class ObservationsCfg(G1ObservationsCfg):
    manipulation = ManipulationCfg()
    object_privileged = ObjectPrivilegedCfg()


@configclass
class EventsCfg(G1EventCfg):
    reset_box = EventTermCfg(func=mdp.reset_box, mode="reset")


@configclass
class RewardsCfg(G1RewardsCfg):
    hand_position = RewardTermCfg(func=mdp.hand_position_tracking, weight=5.0)
    hand_orientation = RewardTermCfg(func=mdp.hand_orientation_tracking, weight=1.0)
    object_velocity = RewardTermCfg(func=mdp.box_velocity_tracking, weight=3.0)


@configclass
class TerminationsCfg(G1TerminationsCfg):
    box_tilted = TerminationTermCfg(func=mdp.box_tilted)
    box_outside_patch = TerminationTermCfg(func=mdp.box_outside_patch)


@configclass
class G1ObjectPushEnvCfg(G1MovingPatchEnvCfg):
    scene: G1ObjectPushSceneCfg = G1ObjectPushSceneCfg(num_envs=16, env_spacing=5.0)
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = G1PhysicsProxyCfg()
        # MJWarp resolves robot-box collisions; MPM receives box and foot proxy colliders.
        self.sim.physics.solver_cfg.entries[0].bodies.append(r"/World/envs/env_.*/Box")
        self.sim.physics.solver_cfg.proxies[0].bodies.append(r"/World/envs/env_.*/Box")
        self.mpm_leaf_node_count_per_world = 1 << 15
        self.mpm_lower_node_count_per_world = 1 << 12
        self.mpm_upper_node_count_per_world = 64
        self.episode_length_s = 12.0
        # Let --video select its renderer instead of inheriting a GL-only recorder.
        self.video_recorders = []
        # Initial task stage: forward pushing and stopping, with reachable box-attached targets.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.debug_vis = False
        for name, value in vars(self.curriculum).items():
            if not name.startswith("_"):
                setattr(self.curriculum, name, None)
        for name in (
            "physics_material",
            "add_base_mass",
            "push_robot",
            "scale_actuator_gains",
            "mpm_material",
            "mpm_material_log",
        ):
            setattr(self.events, name, None)
        self.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.observations.policy.projected_gravity.func = core_mdp.projected_gravity
        self.observations.policy.projected_gravity.params = {}
        # Retain leg/waist posture control without penalizing the arms for reaching.
        params = self.rewards.joint_deviation.params
        params["weight_standing"] = {".*(hip|knee|ankle|waist).*": 0.5, ".*(shoulder|elbow|wrist).*": 0.0}
        for name in ("weight_walking", "weight_running"):
            for pattern in params[name]:
                if any(joint in pattern for joint in ("shoulder", "elbow", "wrist")):
                    params[name][pattern] = 0.0
        for view in self.sim.visualizer_cfgs:
            view.origin_type = "asset"
            view.origin_track_path = "robot"
            view.eye = (2.0, -3.0, 1.5)
            view.lookat = (0.5, 0.0, -0.2)


@configclass
class G1ObjectPushEnvCfg_PLAY(G1ObjectPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_box.params = {"distance_range": (0.9, 0.9)}
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.25)
        self.commands.base_velocity.rel_standing_envs = 0.0
