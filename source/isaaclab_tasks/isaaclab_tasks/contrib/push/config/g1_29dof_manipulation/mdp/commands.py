# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrist pose commands expressed in a configurable robot link frame."""

import numpy as np
import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error


class LinkPoseCommand(UniformPoseCommand):
    """Sample poses in a moving reference link, retaining uniform-pose visualization and metrics."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        body_ids, _ = self.robot.find_bodies(cfg.reference_body_name)
        if len(body_ids) != 1:
            raise ValueError("reference_body_name must select exactly one body.")
        self.reference_body_idx = body_ids[0]

    def tracking_error(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current position [m] and orientation [rad] errors in the commanded link frame."""
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.body_link_pos_w.torch[:, self.reference_body_idx],
            self.robot.data.body_link_quat_w.torch[:, self.reference_body_idx],
            self.command[:, :3],
            self.command[:, 3:],
        )
        position_error, rotation_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_link_pos_w.torch[:, self.body_idx],
            self.robot.data.body_link_quat_w.torch[:, self.body_idx],
        )
        return position_error.norm(dim=-1), rotation_error.norm(dim=-1)

    def _compute_error(self):
        return self.tracking_error()

    def _update_command(self):
        # Refresh markers after resampling, including partial resets.
        self.tracking_error()


class DatasetPoseCommand(LinkPoseCommand):
    """Independently sample validated wrist endpoints on this term's own timer.

    Each arm and environment draws its own dataset row. Independently combined
    arm targets and paths between targets have not been collision-checked.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        with np.load(cfg.dataset_path, allow_pickle=False) as dataset:
            if str(dataset["reference_body_name"].item()) != cfg.reference_body_name:
                raise ValueError("Pose dataset reference frame does not match the command configuration.")
            poses = np.array(dataset[cfg.pose_key], dtype=np.float32, copy=True)
        if poses.ndim != 2 or poses.shape[1] != 7 or poses.shape[0] == 0:
            raise ValueError("Pose dataset must contain a nonempty (N, 7) pose array.")
        if not np.isfinite(poses).all() or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-4):
            raise ValueError("Pose dataset must contain finite positions and unit XYZW quaternions.")
        self._poses = torch.as_tensor(poses, device=self.device)
        self.dataset_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.pose_command_b[:] = self._poses[0]

    def _resample_command(self, env_ids):
        self.dataset_indices[env_ids] = torch.randint(
            self._poses.shape[0], self.dataset_indices[env_ids].shape, device=self.device
        )
        self.pose_command_b[env_ids] = self._poses[self.dataset_indices[env_ids]]
