# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed sand testbeds for the unified G1 MPM task; every spawned particle is simulated."""

from __future__ import annotations

import math

from isaaclab_visualizers.newton import NewtonGLBackendCfg

import isaaclab.envs.mdp as mdp
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

from ...rtx_visualizer import InteractiveRTXBackendCfg
from ...util.solver_setting import configure_sparse_mpm_capacities, get_mpm_solver_cfg
from ...util.testbed import TestbedMotionCfg
from . import mdp as mpm_mdp
from .env_cfg import (
    G1DownhillTestbedSceneCfg,
    G1DuneTestbedSceneCfg,
    G1PhysicsProxyCfg,
    G1TestbedSceneCfg,
    G1TiltingTestbedSceneCfg,
    G1UphillTestbedSceneCfg,
    G1WalkingTiltingTestbedSceneCfg,
)
from .streaming_mpm_env_cfg import G1MPMEnvCfg

"""
fixed testbed with specific surface geometry.
"""


@configclass
class G1TestbedEnvCfg(G1MPMEnvCfg):
    """Velocity-tracking locomotion of the G1 on a fixed sand bed.

    The MDP is the unified task's. Terms that read the shared terrain use the flat bed instead.
    """

    scene: G1TestbedSceneCfg = G1TestbedSceneCfg(num_envs=32, env_spacing=8.0, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        use_newton_actuators=True,
        physics=G1PhysicsProxyCfg(),  # type: ignore
    )

    mpm_activity_colors: bool = True
    mpm_cache_initial_state: bool = True

    def validate_config(self) -> None:
        """Size the sparse storage after CLI/Hydra overrides such as the environment count."""
        if get_mpm_solver_cfg(self).streaming is not None:
            raise ValueError("A fixed testbed simulates every spawned particle; MPMSolverCfg.streaming must be None.")
        for proxy in self.sim.physics.solver_cfg.proxies:
            proxy.mass_scale = self.proxy_mass_scale
        configure_sparse_mpm_capacities(self)

    def __post_init__(self) -> None:
        self.episode_length_s = 13.0

        # The testbeds have no shared terrain: heights and foot pitch refer to the flat initial
        # sand surface, and the workspace is the bed.
        self.curriculum.terrain_levels = None  # type: ignore
        self.rewards.base_height.params = {"target_height": 0.75}
        self.rewards.stance_foot_angle_penalty.params["height_sensor_cfg"] = None
        self.rewards.foot_clearance.params["height_sensor_cfg"] = None
        self.rewards.foot_clearance.params["ground_height_offset"] = 0.0
        self.terminations.terrain_out_of_bounds.func = mpm_mdp.root_outside_sand_bed
        self.terminations.terrain_out_of_bounds.params = {"margin": 0.3, "asset_cfg": SceneEntityCfg("robot")}

        # The robot starts on the sand near the negative-X end of the bed, so the yaw spread is narrow
        # enough that a forward command carries it along the bed rather than off its side.
        self.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.6, 0.6)}

        # The bed is a few metres across, so the high-speed stages of the cloned command
        # curriculum would drive the robot off it within an episode.
        self.curriculum.command_vel = None  # type: ignore
        # The robot spawns near the negative-X end and the bed lies ahead of it, so the command is
        # biased forward; a backward command would walk it off the end of the bed.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.sim.visualizer_cfgs = [
            VisualizerCfg(
                backend=MultiBackendVisualizerCfg(  # type: ignore
                    newton_gl=NewtonGLBackendCfg(show_particles=True),
                    newton_rtx=InteractiveRTXBackendCfg(),
                ),
                eye=(0.0, -6.0, 1.5),
                headless=True,
            )
        ]

        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1TestbedEnvCfg_PLAY(G1TestbedEnvCfg):
    """Small playback scene with randomization disabled."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.mpm_activity_colors = False

        # disable curriculum and observation corruption
        self.curriculum.track_lin_vel = None  # type: ignore
        self.curriculum.track_ang_vel = None  # type: ignore
        self.curriculum.track_heading = None  # type: ignore
        self.curriculum.track_lin_vel_weight = None  # type: ignore
        self.curriculum.gravity_roll_bias = None  # type: ignore
        self.curriculum.gravity_pitch_bias = None  # type: ignore
        self.observations.policy.enable_corruption = False
        self.observations.policy.projected_gravity.func = mdp.projected_gravity
        self.observations.policy.projected_gravity.params = {}

        # remove random events
        self.events.mpm_material = None  # type: ignore
        self.events.mpm_material_log = None  # type: ignore
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)

        self.sim.visualizer_cfgs = [
            VisualizerCfg(
                backend=MultiBackendVisualizerCfg(  # type: ignore
                    newton_gl=NewtonGLBackendCfg(),
                    newton_rtx=InteractiveRTXBackendCfg(),
                ),
                eye=(-2.0, -0.5, 1.0),
                lookat=(0.0, 0.0, 0.0),
                streaming_view=False,
            )
        ]

        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1UphillTestbedEnvCfg(G1TestbedEnvCfg):
    """G1 walking on a fixed generated sand surface.

    This geometry variant retains the base task's rewards. Height and touchdown
    rewards still reference flat ground; adapt them before rough-terrain training.
    """

    scene: G1UphillTestbedSceneCfg = G1UphillTestbedSceneCfg(num_envs=32, env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        # The spawn height is that of the surface at the spawn position; an X offset would leave it.
        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)


@configclass
class G1UphillTestbedEnvCfg_PLAY(G1TestbedEnvCfg_PLAY):
    """Playback with a fixed generated sand surface and no episode randomization."""

    scene: G1UphillTestbedSceneCfg = G1UphillTestbedSceneCfg(num_envs=1, env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        # The spawn height is that of the surface at the spawn position; an X offset would leave it.
        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)

        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1DownhillTestbedEnvCfg(G1UphillTestbedEnvCfg):
    """Locomotion onto a downhill sand ramp."""

    scene: G1DownhillTestbedSceneCfg = G1DownhillTestbedSceneCfg(num_envs=32, env_spacing=8.0)


@configclass
class G1DownhillTestbedEnvCfg_PLAY(G1UphillTestbedEnvCfg_PLAY):
    """Deterministic playback on a downhill sand ramp."""

    scene: G1DownhillTestbedSceneCfg = G1DownhillTestbedSceneCfg(num_envs=1, env_spacing=8.0)


@configclass
class G1DuneTestbedEnvCfg(G1UphillTestbedEnvCfg):
    """Locomotion over sinusoidal sand ridges."""

    scene: G1DuneTestbedSceneCfg = G1DuneTestbedSceneCfg(num_envs=32, env_spacing=8.0)


@configclass
class G1DuneTestbedEnvCfg_PLAY(G1UphillTestbedEnvCfg_PLAY):
    """Deterministic playback over sinusoidal sand ridges."""

    scene: G1DuneTestbedSceneCfg = G1DuneTestbedSceneCfg(num_envs=1, env_spacing=8.0)


"""
testbed with roll/pitch revolute joints, driven by a slow sinusoidal motion.
"""

ROLL_MAX = math.radians(15.0)
PITCH_MAX = math.radians(15.0)


@configclass
class G1TiltingTestbedEnvCfg(G1TestbedEnvCfg):
    """Balance on a slowly tilting sand tray driven through physical joints."""

    scene: G1TiltingTestbedSceneCfg = G1TiltingTestbedSceneCfg(num_envs=4, env_spacing=8.0)
    testbed_motion: TestbedMotionCfg = TestbedMotionCfg(
        roll_range=(-ROLL_MAX, ROLL_MAX),
        pitch_range=(-PITCH_MAX, PITCH_MAX),
        debug_vis=True,
    )

    def validate_config(self) -> None:
        """Hand the tray to both solvers of the selected physics preset, then size the sparse storage."""
        # The same dynamic tray collides with G1 in MJWarp and with particles via its MPM proxy.
        solver_cfg = self.sim.physics.solver_cfg
        rigid, sand = solver_cfg.entries
        for bodies, tray in (
            (rigid.bodies, r"/World/envs/env_.*/Testbed"),
            (solver_cfg.proxies[0].bodies, r"/World/envs/env_.*/Testbed/tray"),
        ):
            if tray not in bodies:
                bodies.append(tray)
        rigid.include_static_shapes = False
        sand.bodies = []
        super().validate_config()

    def __post_init__(self):
        super().__post_init__()
        self.testbed_motion.validate()
        self.episode_length_s = 30.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {}
        self.terminations.terrain_out_of_bounds.func = mpm_mdp.root_outside_testbed
        self.terminations.terrain_out_of_bounds.params = {"margin": 0.3}
        self.rewards.base_height.func = mpm_mdp.base_height_above_testbed


@configclass
class G1TiltingTestbedEnvCfg_PLAY(G1TiltingTestbedEnvCfg):
    """Deterministic balancing evaluation on the tilting sand tray."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.observations.policy.projected_gravity.func = mdp.projected_gravity
        self.observations.policy.projected_gravity.params = {}

        self.mpm_activity_colors = False

        self.events.mpm_material = None
        self.events.mpm_material_log = None
        self.events.add_base_mass = None
        self.events.physics_material = None
        self.events.scale_actuator_gains = None
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        self.commands.base_velocity.debug_vis = False

        self.curriculum.track_lin_vel = None
        self.curriculum.track_ang_vel = None
        self.curriculum.track_heading = None
        self.curriculum.track_lin_vel_weight = None
        self.curriculum.gravity_roll_bias = None
        self.curriculum.gravity_pitch_bias = None
        for visualizer in self.sim.visualizer_cfgs:
            visualizer.headless = False

        self.video_recorders = [
            # VideoRecorderCfg(source="visualizer:newton_gl", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:newton_rtx", output_dir=None, video_length=200, video_interval=2000),
            # VideoRecorderCfg(source="visualizer:kit", output_dir=None, video_length=200, video_interval=2000),
        ]


@configclass
class G1WalkingTiltingTestbedEnvCfg(G1TiltingTestbedEnvCfg):
    """Track planar velocity and yaw-rate commands on a square tilting sand tray."""

    scene: G1WalkingTiltingTestbedSceneCfg = G1WalkingTiltingTestbedSceneCfg(num_envs=4, env_spacing=8.0)
    testbed_motion: TestbedMotionCfg = TestbedMotionCfg(
        roll_range=(math.radians(-10.0), math.radians(10.0)),
        pitch_range=(math.radians(-10.0), math.radians(10.0)),
        debug_vis=True,
    )
    mpm_leaf_node_count_per_world: int = 1 << 15
    mpm_lower_node_count_per_world: int = 1 << 12
    mpm_upper_node_count_per_world: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.0


@configclass
class G1WalkingTiltingTestbedEnvCfg_PLAY(G1TiltingTestbedEnvCfg_PLAY):
    """Forward-walking playback; command ranges can select lateral walking or turning."""

    scene: G1WalkingTiltingTestbedSceneCfg = G1WalkingTiltingTestbedSceneCfg(num_envs=1, env_spacing=8.0)
    testbed_motion: TestbedMotionCfg = TestbedMotionCfg(debug_vis=True)
    mpm_leaf_node_count_per_world: int = 1 << 15
    mpm_lower_node_count_per_world: int = 1 << 12
    mpm_upper_node_count_per_world: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.debug_vis = True
