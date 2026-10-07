# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Standalone G1 terminal-goal navigation and whole-body posture environment."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ...mdp import pose_goal_observations as observations
from ...mdp import pose_goal_rewards as rewards
from ...mdp import pose_goal_symmetry as symmetry
from ...mdp.pose_goal_commands_cfg import PoseGoalCommandCfg
from ...mdp.pose_goal_events import reset_pose_goal_robot
from ...mdp.terminations import fallen
from .env_cfg.physics_cfg import G1PhysicsCfg
from .env_cfg.scene_cfg import G1SceneCfg
from .robot_constants import JOINT_NAMES

ObsTerm = symmetry.MirrorObservationTermCfg
ROBOT_JOINTS = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)


@configclass
class G1PoseGoalSceneCfg(G1SceneCfg):
    """The existing box-foot robot and flat plane, without a joint-reference ghost."""

    ghost = None


@configclass
class G1PoseGoalCommandsCfg:
    pose_goal = PoseGoalCommandCfg(joint_names=JOINT_NAMES)


@configclass
class G1PoseGoalActionsCfg:
    joint_position = symmetry.MirrorJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
        scale=0.2,
        use_default_offset=True,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": JOINT_NAMES},
    )


@configclass
class G1PoseGoalObservationsCfg:
    """Deployable ideal-state actor and force-privileged critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=symmetry.mirror_vec3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, mirror=symmetry.mirror_vec3, mirror_params={"axial": True})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, mirror=symmetry.mirror_vec3)
        torso_orientation = ObsTerm(func=observations.torso_orientation_p, mirror=symmetry.mirror_rotation_6d)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": ROBOT_JOINTS},
            mirror=symmetry.mirror_g1_joints,
            mirror_params={"joint_names": JOINT_NAMES},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": ROBOT_JOINTS},
            scale=0.05,
            mirror=symmetry.mirror_g1_joints,
            mirror_params={"joint_names": JOINT_NAMES},
        )
        hand_positions = ObsTerm(func=observations.hand_positions_t, mirror=symmetry.mirror_hand_positions)
        hand_orientations = ObsTerm(func=observations.hand_orientations_t, mirror=symmetry.mirror_hand_orientations)

        last_action = ObsTerm(
            func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": JOINT_NAMES}
        )
        goal = ObsTerm(func=observations.pose_goals, mirror=symmetry.mirror_goal_command)

    @configclass
    class CriticCfg(PolicyCfg):
        contacts = ObsTerm(func=observations.foot_contacts, mirror=symmetry.mirror_foot_scalars)
        clearance = ObsTerm(func=observations.foot_clearance, mirror=symmetry.mirror_foot_scalars)
        forces = ObsTerm(func=observations.foot_forces, scale=0.01, mirror=symmetry.mirror_foot_forces)

    policy = PolicyCfg(concatenate_terms=True, enable_corruption=False)
    critic = CriticCfg(concatenate_terms=True, enable_corruption=False)


@configclass
class G1PoseGoalRewardsCfg:
    """Terminal task rewards with no foot-pose or target-joint tracking."""

    pelvis_distance = RewTerm(func=rewards.pelvis_distance, weight=4.0)
    pelvis_position = RewTerm(func=rewards.pelvis_position, weight=2.0)
    pelvis_height = RewTerm(func=rewards.pelvis_height, weight=1.0)
    pelvis_yaw = RewTerm(func=rewards.pelvis_yaw, weight=1.5)
    torso_orientation = RewTerm(func=rewards.torso_orientation, weight=1.0)
    hand_position = RewTerm(func=rewards.hand_position, weight=2.0)
    hand_orientation = RewTerm(func=rewards.hand_orientation, weight=0.5)
    settled_hold = RewTerm(func=rewards.settled_hold, weight=2.0)
    foot_clearance = RewTerm(
        func=rewards.foot_clearance,
        weight=0.5,
        params={"clearance_height": 0.05, "standard_deviation": 0.03, "speed_scale": 0.5},
    )
    arrival_velocity = RewTerm(func=rewards.arrival_velocity, weight=-0.5)
    pelvis_tilt = RewTerm(func=rewards.pelvis_tilt, weight=-0.5)
    foot_sliding = RewTerm(func=rewards.foot_sliding, weight=-0.2)
    foot_scuffing = RewTerm(func=rewards.foot_scuffing, weight=-0.1)
    no_support = RewTerm(func=rewards.no_support, weight=-2.0)
    excessive_foot_force = RewTerm(func=rewards.excessive_foot_force, weight=-0.1)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class G1PoseGoalEventsCfg:
    reset_robot = EventTermCfg(func=reset_pose_goal_robot, mode="reset")


@configclass
class G1PoseGoalTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fallen = TerminationTermCfg(func=fallen)


@configclass
class G1PoseGoalEnvCfg(ManagerBasedRLEnvCfg):
    """WBC 2.0 on Newton MJWarp, 60 Hz policy with terminal goals."""

    scene = G1PoseGoalSceneCfg(num_envs=1024, env_spacing=4.0)
    actions = G1PoseGoalActionsCfg()
    observations = G1PoseGoalObservationsCfg()
    commands = G1PoseGoalCommandsCfg()
    rewards = G1PoseGoalRewardsCfg()
    events = G1PoseGoalEventsCfg()
    terminations = G1PoseGoalTerminationsCfg()
    sim = SimulationCfg(dt=1 / 240, physics=G1PhysicsCfg())
    decimation = 4
    episode_length_s = 20.0

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.visualizer_cfgs = []

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.0, 0.0, 0.8), headless=True),
            NewtonRTXVisualizerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.0, 0.0, 0.8), headless=True),
        ]
        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000)
        ]


@configclass
class G1PoseGoalEnvCfgPlay(G1PoseGoalEnvCfg):
    """Evaluate the final command range without an automatic training curriculum."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.pose_goal.curriculum_steps = 0
        self.commands.pose_goal.debug_vis = True

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.0, 0.0, 0.8), headless=True),
            NewtonRTXVisualizerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.0, 0.0, 0.8), headless=True),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000)
        ]