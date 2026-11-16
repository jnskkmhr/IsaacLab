# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compositional RSL-RL PPO configuration for the `box` two-way coupling task."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import BoxParticleNeRDLocalPPORunnerCfg


@configclass
class BoxParticleNeRDLocalCompositionalPPORunnerCfg(BoxParticleNeRDLocalPPORunnerCfg):
    experiment_name = "particle_nerd_local_box_compositional"
