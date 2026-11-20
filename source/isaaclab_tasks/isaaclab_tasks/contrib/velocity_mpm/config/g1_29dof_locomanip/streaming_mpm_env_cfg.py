# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug configuration: the streaming G1 MPM task with a free box, sphere and cylinder in front of the robot."""

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..g1_29dof.streaming_mpm_env_cfg import G1MPMEnvCfg, G1MPMEnvCfg_PLAY
from .env_cfg import G1LocomanipEventCfg, G1LocomanipPhysicsCfg, G1LocomanipSceneCfg


@configclass
class G1LocomanipEnvCfg(G1MPMEnvCfg):
    """The streaming task with three free objects; the robot and each object carry a streaming pool."""

    scene: G1LocomanipSceneCfg = G1LocomanipSceneCfg(num_envs=1, env_spacing=4.0)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        use_newton_actuators=True,
        physics=G1LocomanipPhysicsCfg(),  # type: ignore
    )
    events: G1LocomanipEventCfg = G1LocomanipEventCfg()


@configclass
class G1LocomanipEnvCfg_PLAY(G1MPMEnvCfg_PLAY):
    """Playback in which the robot starts facing the box and walks straight into it."""

    scene: G1LocomanipSceneCfg = G1LocomanipSceneCfg(num_envs=1, env_spacing=4.0)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        use_newton_actuators=True,
        physics=G1LocomanipPhysicsCfg(),  # type: ignore
    )
    events: G1LocomanipEventCfg = G1LocomanipEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.mpm_material = None
        self.events.mpm_material_log = None
