# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for the `particle` task: a sand block dropped or tossed onto open flat ground."""

import gymnasium as gym

_MODULE = f"{__name__}.particle_env_cfg"

for _shape in ("Cube", "Sphere", "Cylinder"):
    for _motion in ("Drop", "Throw"):
        for _suffix, _class_suffix in (("", ""), ("-MPM", "MPM")):
            gym.register(
                id=f"IsaacContrib-ParticleNeRD-Particle-{_shape}-{_motion}{_suffix}-v0",
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={"env_cfg_entry_point": f"{_MODULE}:Particle{_shape}{_motion}{_class_suffix}EnvCfg"},
            )
