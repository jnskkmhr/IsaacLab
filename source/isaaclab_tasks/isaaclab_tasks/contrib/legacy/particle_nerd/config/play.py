# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint-free interactive playback; the scene supplies the particle checkpoint."""


def run(argv=None):
    """Apply zero external action while gravity, material reactions and mouse forces act."""
    from isaaclab_rl.entrypoints import run_zero_agent_cli

    return run_zero_agent_cli(argv)
