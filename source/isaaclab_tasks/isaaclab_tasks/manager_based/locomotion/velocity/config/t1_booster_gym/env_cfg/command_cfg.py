from __future__ import annotations

from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_booster_gym.mdp as t1_mdp


@configclass
class T1CommandsCfg:
    base_velocity = t1_mdp.UniformVelocityCommandPlsCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        heading_command=False,
        debug_vis=True,
        ranges=t1_mdp.UniformVelocityCommandPlsCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),  # type: ignore
            lin_vel_y=(-1.0, 1.0),  # type: ignore
            ang_vel_z=(-1.0, 1.0),  # type: ignore
        ),
        rel_standing_envs=0.15,
        rel_rotonly_envs=0.15,
    )

    phase = t1_mdp.PhaseCommandCfg(
        resampling_time_range=(1e10, 1e10),  # resample on command resample
        phase_frequency_hz_range=[1.0, 2.0]
    )
