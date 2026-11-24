# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based G1 configurations for towing a sled over mixed rigid ground and analytical (RFT) sand."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import RigidBodyMaterialCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab.visualizers.visualizer_cfg import WindowCfg

from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1ObservationsCfg,
    G1PhysicsCfg,
    G1RewardsCfg,
    G1TerminationsCfg,
    G1TowSceneCfg,
    TetherCfg,
)


@configclass
class G1TowEnvCfg(ManagerBasedRLEnvCfg):
    """G1 walks on rigid ground and tows a sled by a rope tied to its pelvis."""

    scene = G1TowSceneCfg(num_envs=4096, env_spacing=4.0)
    actions = G1ActionsCfg()
    observations = G1ObservationsCfg()
    commands = G1CommandsCfg()
    events = G1EventCfg()
    rewards = G1RewardsCfg()
    curriculum = G1CurriculumCfg()
    terminations = G1TerminationsCfg()
    video_recorders = [VideoRecorderCfg(source="visualizer:newton_gl", video_length=200, video_interval=2000)]
    seed = 42
    tether: TetherCfg = TetherCfg()
    """The rope between the robot and the sled."""
    camera_follow_eye: tuple[float, float, float] | None = (3.0, 3.0, 2.0)
    """Viewer position relative to the last robot's root [m]; the viewer looks at the root. None keeps a fixed view."""
    decimation = 4
    episode_length_s = 20.0
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=G1PhysicsCfg(),
        visualizer_cfgs=[
            # One shared view; the backend is chosen on the command line (``--viz newton_gl`` or ``newton_rtx``).
            VisualizerCfg(
                headless=True,
                window=WindowCfg(size=(1280, 720), fps=30.0),
                focal_length=12.0,
                streaming_view=False,
                eye=(4.0, -4.0, 3.0),
                lookat=(0.0, 0.0, 0.0),
            ),
        ],
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


@configclass
class G1TowEnvCfg_PLAY(G1TowEnvCfg):
    """One robot, deterministic reset and a straight forward command for inspecting the tow."""

    video_recorders = []

    def __post_init__(self):
        super().__post_init__()
        # One rigid-ground and one sand environment.
        self.scene.num_envs = 2
        for visualizer in self.sim.visualizer_cfgs:
            visualizer.headless = False
        for name in ("add_base_mass", "push_robot", "physics_material", "scale_actuator_gains"):
            setattr(self.events, name, None)
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
