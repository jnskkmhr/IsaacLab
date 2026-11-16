# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Play a registered benchmark with its prescribed, IK, or learned controller."""


def run(argv=None):
    """Serve the task's ``isaaclab play`` entrypoint."""
    from .run import run as run_benchmark

    return run_benchmark(argv, interactive=True)
