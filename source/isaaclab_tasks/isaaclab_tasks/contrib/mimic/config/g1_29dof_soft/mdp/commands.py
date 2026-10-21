# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion commands with stance intervals for soft-contact G1 tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.mdp.commands import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SoftMotionCommand(MotionCommand):
    """Extend reference tracking with soft-task stance weights and metrics."""

    cfg: SoftMotionCommandCfg

    def __init__(self, cfg: SoftMotionCommandCfg, env: ManagerBasedRLEnv):
        if not math.isfinite(cfg.stance_blend_time) or cfg.stance_blend_time < 0.0:
            raise ValueError("stance_blend_time must be finite and nonnegative.")
        for start, end in cfg.stance_phase_ranges:
            if not 0.0 <= start <= end <= 1.0:
                raise ValueError("Stance intervals must satisfy 0 <= start <= end <= 1.")
        super().__init__(cfg, env)
        self.frame_stance_weight = self._build_frame_stance_weight()
        self.metrics["standing_weight"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def standing_weight(self) -> torch.Tensor:
        """Blended stance weight for each environment's current reference frame."""
        return self.frame_stance_weight[self.reference_frame_indices]

    @property
    def tracking_weight(self) -> torch.Tensor:
        """Complement of the stance weight."""
        return 1.0 - self.standing_weight

    @property
    def is_stance(self) -> torch.Tensor:
        """Whether the reference has a stance weight greater than 0.5."""
        return self.standing_weight > 0.5

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["standing_weight"][:] = self.standing_weight

    def _build_frame_stance_weight(self) -> torch.Tensor:
        """Smooth stance intervals independently within each reference clip."""
        weights = torch.zeros(self.motion.time_step_total, device=self.device)
        blend_frames = int(round(self.cfg.stance_blend_time * self.motion.fps))
        for start, end in self.motion.frame_ranges:
            phase = torch.arange(end - start, device=self.device).float() / max(end - start - 1, 1)
            weight = torch.zeros_like(phase)
            for phase_start, phase_end in self.cfg.stance_phase_ranges:
                weight[(phase >= phase_start) & (phase <= phase_end)] = 1.0
            if blend_frames > 0:
                width = 2 * blend_frames + 1
                kernel = torch.ones(1, 1, width, device=self.device) / width
                padded = torch.nn.functional.pad(weight.view(1, 1, -1), (blend_frames, blend_frames), mode="replicate")
                weight = torch.nn.functional.conv1d(padded, kernel).view(-1)
            weights[start:end] = weight
        return weights


@configclass
class SoftMotionCommandCfg(MotionCommandCfg):
    """Reference tracking with optional stance phases for soft-contact tasks."""

    class_type: type = SoftMotionCommand

    stance_phase_ranges: list[tuple[float, float]] = []
    """Inclusive normalized stance intervals, applied independently to each clip."""

    stance_blend_time: float = 0.2
    """Box-filter half-width in seconds for smoothing stance transitions."""
