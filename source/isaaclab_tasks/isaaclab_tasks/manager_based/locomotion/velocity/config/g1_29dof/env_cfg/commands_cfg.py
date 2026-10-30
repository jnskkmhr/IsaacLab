# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp


@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = vel_mdp.UniformLevelVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.2,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=vel_mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)
    #     ),
    # )

    base_velocity = vel_mdp.UniformVelocityYawCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=vel_mdp.UniformVelocityYawCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )

    foot_height = vel_mdp.SwingCommandCfg(
        # foot_height=(0.08, 0.15),
        foot_height=(0.1, 0.1),
        resampling_time_range=(1e10, 1e10),  # resample on command resample
    )

    # locomotion_gait = vel_mdp.PhaseCommandCfg(
    #     resampling_time_range=(1e10, 1e10),  # resample on command resample
    #     gait_period=(0.2 * 2, 0.2 * 2),
    #     # gait_period=(0.5 * 2, 0.5 * 2),
    #     sampler="uniform",
    #     ss_duration_phase=0.45,
    #     ds_duration_phase=0.05,
    #     ss_duration_phase_running=0.4,
    #     flight_duration_phase_running=0.1,
    # )

    # locomotion_gait = vel_mdp.PhaseCommandSSPCfg(
    #     resampling_time_range=(1e10, 1e10),  # resample on command resample
    #     gait_period=(0.5 * 2, 0.5 * 2),
    #     sampler="uniform",
    #     ss_duration_phase=0.5,
    #     debug_vis=True,
    #     asset_name="robot",
    #     body_names=[".*ankle_roll.*"],
    # )

    # locomotion_gait = vel_mdp.PhaseCommandDSPCfg(
    #     resampling_time_range=(1e10, 1e10),  # resample on command resample
    #     # gait_period=(0.2 * 2, 0.2 * 2),
    #     gait_period=(0.5 * 2, 0.5 * 2),
    #     sampler="uniform",
    #     ss_duration_phase=0.45,
    #     ds_duration_phase=0.05,
    #     debug_vis=True,
    #     asset_name="robot",
    #     body_names=[".*ankle_roll.*"],
    # )

    # locomotion_gait = vel_mdp.PhaseCommandFLTCfg(
    #     asset_name="robot",
    #     body_names=[".*ankle_roll.*"],
    #     resampling_time_range=(1e10, 1e10),  # resample on command resample
    #     gait_period=(0.2 * 2, 0.2 * 2),
    #     sampler="uniform",
    #     ss_duration_phase=0.5,
    #     max_flt_duration_phase=0.15,
    #     velocity_flight_threshold=1.5,
    #     velocity_flight_maximum=2.5,
    #     debug_vis=True,
    # )
