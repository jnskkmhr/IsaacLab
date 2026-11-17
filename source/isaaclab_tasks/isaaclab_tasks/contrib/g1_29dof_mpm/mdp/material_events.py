# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Per-environment material randomization ported from the source task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp
from isaaclab_newton import assets

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

from .material_access import MPMMaterialAccess

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_mpm_material(ManagerTermBase):
    """Sample one MPM material parameter set per environment for the selected particle object.

    ``material_parameters`` stores the latest per-environment samples on the simulation device,
    for logging or privileged observations. Density samples describe the requested material density,
    including for particles whose simulation mass is zero.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._materials = MPMMaterialAccess(env.sim.backend_manager)
        asset = env.scene[cfg.params["asset_cfg"].name]
        if not isinstance(asset, assets.MPMObject):
            raise TypeError("randomize_mpm_material requires an MPMObject asset.")
        if asset.num_instances != env.num_envs:
            raise ValueError("MPM material randomization requires one particle object per environment.")
        self._particle_offsets = wp.to_torch(asset._particle_offsets).long()
        self._particle_indices = torch.arange(asset._particles_per_object, device=env.device)
        self._env_ids = torch.arange(env.num_envs, device=env.device)
        # Event terms are constructed before the solver is available. Read defaults on first use.
        self.material_parameters: dict[str, torch.Tensor] = {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        parameter_ranges: dict[str, tuple[float, float]],
        distribution: Literal["uniform", "log_uniform"] = "uniform",
    ) -> None:
        """Apply absolute material samples to the selected environments before their next physics step."""
        del env, asset_cfg
        ids = self._env_ids[slice(None) if env_ids is None else env_ids]
        if ids.numel() == 0 or not parameter_ranges:
            return
        if distribution not in ("uniform", "log_uniform"):
            raise ValueError(f"Unsupported MPM material distribution: {distribution}")
        sample = math_utils.sample_uniform if distribution == "uniform" else math_utils.sample_log_uniform
        values = {}
        samples = {}
        for name, (low, high) in parameter_ranges.items():
            if name not in self.material_parameters:
                default_parameters = self._materials.get_particle_material_parameters(
                    wp.from_torch(self._particle_offsets), parameters=(name,)
                )
                self.material_parameters[name] = default_parameters[name].torch
            if low > high or (distribution == "log_uniform" and low <= 0.0):
                raise ValueError(f"Invalid {distribution} range for {name}: {(low, high)}")
            samples[name] = sample(low, high, (ids.numel(),), device=ids.device)
            values[name] = samples[name][:, None].expand(-1, self._particle_indices.numel()).reshape(-1)
        particle_ids = (self._particle_offsets[ids, None] + self._particle_indices[None, :]).reshape(-1)
        self._materials.set_particle_material_parameters(
            {name: wp.from_torch(value) for name, value in values.items()}, wp.from_torch(particle_ids)
        )
        for name, value in samples.items():
            self.material_parameters[name][ids] = value
