# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from particle_nerd_benchmark.core.physics import physics_cfg

OBJECT_BODIES = [r"/World/envs/env_.*/Object"]


def object_physics_cfg():
    return physics_cfg(OBJECT_BODIES, OBJECT_BODIES)
