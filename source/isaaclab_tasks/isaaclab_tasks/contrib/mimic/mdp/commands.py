# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

"""Common functions containing command generators for whole body tracking."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    convert_quat,
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        quaternion_order: Literal["wxyz", "xyzw"] = "wxyz",
        source_joint_names: Sequence[str] | None = None,
        target_joint_names: Sequence[str] | None = None,
    ):
        """Load motion with runtime quaternions in xyzw order.

        NPZ ``quaternion_order`` metadata takes precedence over the fallback argument.
        Untagged legacy motion files default to wxyz; untagged xyzw files must opt in.
        Explicit source/target joint names reorder both joint position and velocity.
        Omitting both preserves the stored joint column order.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file, allow_pickle=False)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        if "quaternion_order" in data:
            quaternion_order = str(np.asarray(data["quaternion_order"]).item())
        if quaternion_order not in ("wxyz", "xyzw"):
            data.close()
            raise ValueError(f"Unsupported motion quaternion order: {quaternion_order!r}")
        if quaternion_order == "wxyz":
            self._body_quat_w = convert_quat(self._body_quat_w, to="xyzw")
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        data.close()
        if (source_joint_names is None) != (target_joint_names is None):
            raise ValueError("Both source_joint_names and target_joint_names must be provided together.")
        if source_joint_names is not None:
            source_names = list(source_joint_names)
            target_names = list(target_joint_names)
            if self.joint_pos.ndim != 2 or self.joint_vel.shape != self.joint_pos.shape:
                raise ValueError("Motion joint position and velocity must have matching [frames, joints] shapes.")
            if (
                len(source_names) != self.joint_pos.shape[1]
                or len(set(source_names)) != len(source_names)
                or len(set(target_names)) != len(target_names)
                or set(source_names) != set(target_names)
            ):
                raise ValueError("Motion and robot joint names must uniquely cover the same joint columns.")
            joint_indexes = torch.tensor([source_names.index(name) for name in target_names], device=device)
            self.joint_pos = self.joint_pos[:, joint_indexes]
            self.joint_vel = self.joint_vel[:, joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # The motion file stores exactly `len(cfg.body_names)` bodies, already in that list's order --
        # not the robot's own (larger) body numbering. Index it by position, not by robot body index,
        # or lookups for any tracked body whose robot index exceeds the motion file's body count go
        # out of bounds.
        motion_body_indexes = torch.arange(len(self.cfg.body_names), dtype=torch.long, device=self.device)
        self.motion = MotionLoader(
            self.cfg.motion_file,
            motion_body_indexes,
            device=self.device,
            quaternion_order=self.cfg.motion_quaternion_order,
            source_joint_names=self.cfg.motion_joint_names,
            target_joint_names=self.robot.joint_names if self.cfg.motion_joint_names is not None else None,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_stance_weight = self._build_frame_stance_weight()
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 3] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        # Where episodes *died*, which is what biases the sampler toward the hard parts of the clip.
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)

        # Failure bookkeeping for the assistive-wrench curriculum, keyed on the bin an episode *started*
        # in rather than the one it ended in. The distinction matters: outside the final bin, the only
        # way to end in a bin is to die there, so an end-keyed rate would read 1.0 forever and the wrench
        # would never retire. Every bin gets starts from the sampler, so successes are actually visible.
        # ZEST S6 keys beta on the sampled bin too. The two counts are EMAs over the same window, so
        # their ratio is the failure level f_b however many episodes happen to end on a given step.
        # Guards the start-frame draw so it happens exactly once per reset, whether it is pulled
        # forward by a `reset_*_from_reference` event or falls through to `_resample_command`.
        self._resample_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.start_bins = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._start_bin_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bin_start_failed_count = torch.ones(self.bin_count, dtype=torch.float, device=self.device)
        self.bin_start_count = torch.ones(self.bin_count, dtype=torch.float, device=self.device)
        self._current_start_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_start_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        # Per-env assistive gain beta_e, frozen at reset from the bin the episode starts in (ZEST S6).
        self.assist_scale = torch.full((self.num_envs,), self.cfg.assist_beta_max, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["standing_weight"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["assist_scale"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def bin_failure_rate(self) -> torch.Tensor:
        """Per-bin failure level f_b in [0, 1], smoothed across neighbouring bins (ZEST S6).

        The ratio of two EMAs taken over the same window: episodes *starting* in the bin that went on to
        terminate, over all episodes that started there. Time-outs count as successes, so running the
        reference out to its last frame (``end_of_reference``) is a success, not a failure.
        """
        return self._smooth_bins(self.bin_start_failed_count / self.bin_start_count.clamp(min=1e-6))

    @property
    def bin_assist_scale(self) -> torch.Tensor:
        """Per-bin assistive gain beta_b = clip(1 - s_b / eta, 0, beta_max), with s_b = 1 - f_b.

        Monotone in the failure level and vanishing once a bin's similarity reaches the target
        ``assist_eta``, so the wrench retires itself bin by bin as the policy masters the clip.
        """
        similarity = 1.0 - self.bin_failure_rate
        return torch.clamp(1.0 - similarity / self.cfg.assist_eta, min=0.0, max=self.cfg.assist_beta_max)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        # return torch.cat([self.joint_pos, self.joint_vel], dim=1)
        return self.joint_pos

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def phase(self) -> torch.Tensor:
        """Normalized position in the clip, in [0, 1]: 0 on the first frame, 1 on the last."""
        return self.time_steps.float() / max(self.motion.time_step_total - 1, 1)

    @property
    def has_reached_end(self) -> torch.Tensor:
        """True once the reference is sitting on the last frame of the clip, i.e. it has played out."""
        return self.time_steps >= self.motion.time_step_total - 1

    @property
    def standing_weight(self) -> torch.Tensor:
        """Per-env weight in [0, 1] for standing/balance rewards.

        1 while the reference is inside a configured stance interval, 0 while it is tracking a real
        trajectory, and a linear ramp across ``cfg.stance_blend_time`` seconds on either side of a
        boundary so the reward the policy sees does not step. With no ``stance_phase_ranges``
        configured this is 0 everywhere, i.e. the pre-stance-gating behavior.
        """
        return self.frame_stance_weight[self.time_steps]

    @property
    def tracking_weight(self) -> torch.Tensor:
        """Complement of :attr:`standing_weight`: the weight for motion-tracking rewards."""
        return 1.0 - self.standing_weight

    @property
    def is_stance(self) -> torch.Tensor:
        """True where the reference is more stance than tracking (weight > 0.5)."""
        return self.standing_weight > 0.5

    def _build_frame_stance_weight(self) -> torch.Tensor:
        """Precompute the per-frame stance weight from ``cfg.stance_phase_ranges``.

        The ranges are given in phase units, so they survive a change in clip length (e.g. re-running
        ``edit_longer_stance.py`` with a different hold) as long as the stance stays proportionally
        in the same place. The hard 0/1 mask is smoothed with a box filter whose width is set by
        ``cfg.stance_blend_time``, which turns each boundary into a linear ramp of that duration.
        """
        num_frames = self.motion.time_step_total
        weight = torch.zeros(num_frames, device=self.device)
        if not self.cfg.stance_phase_ranges:
            return weight

        frame_phase = torch.arange(num_frames, device=self.device).float() / max(num_frames - 1, 1)
        for phase_start, phase_end in self.cfg.stance_phase_ranges:
            weight[(frame_phase >= phase_start) & (frame_phase <= phase_end)] = 1.0

        blend_frames = int(round(self.cfg.stance_blend_time * self.motion.fps))
        if blend_frames > 0:
            # a box filter of width 2 * blend_frames + 1 turns every 0 -> 1 edge into a linear ramp
            # spanning blend_frames on each side of the boundary
            kernel = torch.ones(1, 1, 2 * blend_frames + 1, device=self.device) / (2 * blend_frames + 1)
            padded = torch.nn.functional.pad(weight.view(1, 1, -1), (blend_frames, blend_frames), mode="replicate")
            weight = torch.nn.functional.conv1d(padded, kernel).view(-1)
        return weight

    def _update_metrics(self):
        self.metrics["phase"] = self.phase
        self.metrics["standing_weight"] = self.standing_weight
        self.metrics["assist_scale"][:] = self.assist_scale
        self.metrics["failure_rate"][:] = self.bin_failure_rate.mean()
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _smooth_bins(self, values: torch.Tensor) -> torch.Tensor:
        """Blur a per-bin quantity along the clip with the configured non-causal kernel."""
        padded = torch.nn.functional.pad(
            values.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        return torch.nn.functional.conv1d(padded, self.kernel.view(1, 1, -1)).view(-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]  # type: ignore

        # Sampler bookkeeping: bin the ending episodes by where the reference stopped.
        current_bin_index = torch.clamp(
            (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
        )
        self._current_bin_failed[:] = torch.bincount(
            current_bin_index[env_ids][episode_failed], minlength=self.bin_count
        )

        # Assist-curriculum bookkeeping: attribute each outcome to the bin that episode started in.
        # Envs that have not run an episode yet carry no start bin and are skipped, so the very first
        # reset does not book a batch of phantom successes against bin 0.
        valid = self._start_bin_valid[env_ids]
        finished_start_bins = self.start_bins[env_ids][valid]
        self._current_start_count[:] = torch.bincount(finished_start_bins, minlength=self.bin_count)
        self._current_start_failed[:] = torch.bincount(
            finished_start_bins[episode_failed[valid]], minlength=self.bin_count
        )

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = self._smooth_bins(sampling_probabilities)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        if self.cfg.start_from_beginning:
            self.time_steps[env_ids] = 0
        else:
            self.time_steps[env_ids] = (
                (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
                / self.bin_count
                * (self.motion.time_step_total - 1)
            ).long()

        # Freeze the assistive gain for the whole episode at the bin the reference starts in, so the
        # policy sees a constant amount of help rather than a wrench that fades out underneath it.
        start_bins = torch.zeros_like(sampled_bins) if self.cfg.start_from_beginning else sampled_bins
        self.assist_scale[env_ids] = self.bin_assist_scale[start_bins]
        self.start_bins[env_ids] = start_bins
        self._start_bin_valid[env_ids] = True

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def resample_time_steps(self, env_ids: Sequence[int]):
        """Draw the start frame for ``env_ids`` from the adaptive sampler, at most once per reset.

        Placing the robot on the reference is a reset *event*'s job (see
        :func:`mimic.mdp.events.reset_root_state_from_reference` and
        :func:`~mimic.mdp.events.reset_joint_state_from_reference`), but the event manager runs earlier
        in ``_reset_idx`` than the command manager, so those events have to pull this draw forward or
        they would place the robot on the frame the *previous* episode ended at. Calling this is
        therefore safe and idempotent: whichever of the event or :meth:`_resample_command` gets there
        first does the draw, and the other is a no-op.
        """
        env_ids = torch.as_tensor(env_ids, device=self.device)
        pending = env_ids[self._resample_pending[env_ids]]
        if len(pending) == 0:
            return
        self._adaptive_sampling(pending)
        self._resample_pending[pending] = False

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        # Only draws if no reset event already did, then re-arms for the next reset. The robot's own
        # state is written by the `reset_*_from_reference` events, not here.
        self.resample_time_steps(env_ids)
        self._resample_pending[env_ids] = True

    def _update_command(self):
        # Hold at the last frame once the clip is exhausted. Ending the episode is a termination's
        # job (see `mimic.mdp.terminations.end_of_reference`), not this term's: resetting from here
        # would teleport the robot into a fresh clip segment mid-episode, which is invisible to the
        # termination and reward managers. Register `end_of_reference` or the reference will sit
        # frozen on its final frame until the episode times out.
        self.time_steps = torch.clamp(self.time_steps + 1, max=self.motion.time_step_total - 1)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        alpha = self.cfg.adaptive_alpha
        self.bin_failed_count = alpha * self._current_bin_failed + (1 - alpha) * self.bin_failed_count
        self.bin_start_failed_count = alpha * self._current_start_failed + (1 - alpha) * self.bin_start_failed_count
        self.bin_start_count = alpha * self._current_start_count + (1 - alpha) * self.bin_start_count
        self._current_bin_failed.zero_()
        self._current_start_failed.zero_()
        self._current_start_count.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")  # type: ignore
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")  # type: ignore
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)  # type: ignore
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)  # type: ignore
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING  # type: ignore

    motion_file: str = MISSING  # type: ignore
    motion_quaternion_order: Literal["wxyz", "xyzw"] = "wxyz"
    """Quaternion order for untagged NPZ files; file metadata overrides this fallback.

    Legacy references use wxyz. Runtime tensors and newly converted references use xyzw.
    """
    motion_joint_names: list[str] | None = None
    """Joint names in NPZ column order; None assumes the robot already uses that order.

    Set this for the specific motion file, not from the body list. Both position and
    velocity are reordered by name into robot joint order when loading.
    """
    anchor_body_name: str = MISSING  # type: ignore
    body_names: list[str] = MISSING  # type: ignore

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    start_from_beginning: bool = False

    assist_eta: float = 0.8
    """Target similarity at which the assistive wrench retires (``eta`` in ZEST S6).

    A bin gets help only while its smoothed similarity 1 - f_b sits below this, i.e. while its failure
    rate exceeds ``1 - assist_eta``. Larger values keep the wrench around longer."""

    assist_beta_max: float = 0.5
    """Cap on the assistive gain. Kept below 1.0 so assistance stays partial: the policy still has to
    do most of the work, and it does not overfit to physics that will be taken away."""

    stance_phase_ranges: list[tuple[float, float]] = []
    """Intervals of the clip, in phase units ([0, 1], inclusive on both ends), where the reference is
    a static stance rather than a trajectory worth tracking.

    Inside these intervals the motion-tracking rewards are gated off and standing/balance rewards are
    gated on (see :attr:`MotionCommand.standing_weight`). An empty list disables stance gating
    entirely, which reproduces the pure motion-tracking behavior."""

    stance_blend_time: float = 0.2
    """Duration (seconds) of the linear ramp on each side of a stance boundary, so the reward the
    policy sees transitions smoothly instead of stepping. 0.0 gives hard switches."""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")  # type: ignore
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)  # type: ignore

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")  # type: ignore
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore
