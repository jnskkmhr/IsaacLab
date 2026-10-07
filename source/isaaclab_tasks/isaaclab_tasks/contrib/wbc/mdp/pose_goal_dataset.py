# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Endpoint forward kinematics and collision sole points from the task's actual USD."""

from itertools import product

import newton
import numpy as np
import torch
import warp as wp


def endpoint_kinematics(
    usd_path: str, joint_names: list[str], joint_pos: torch.Tensor, root_pose: torch.Tensor, body_names: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate named link poses and box-foot collision corners on CPU.

    Args:
        usd_path: The same robot USD used by the environment.
        joint_names: Joint column order.
        joint_pos: Endpoint joint angles [rad], shape (N, J), on CPU.
        root_pose: World pelvis XYZ and XYZW, shape (N, 7), on CPU.
        body_names: Requested link names; the last two must be the feet.

    Returns:
        Link XYZ/XYZW poses (N, B, 7), and foot-local collision corners (2, 8, 3).
        All corners are retained so minimum height remains correct for rotated feet.
    """
    template = newton.ModelBuilder()
    template.add_usd(usd_path, collapse_fixed_joints=False, load_visual_shapes=False, skip_mesh_approximation=True)
    if template.joint_type[0] != newton.JointType.FREE:
        raise ValueError("WBC endpoints require a floating-base robot")
    bodies = {name.rsplit("/", 1)[-1]: i for i, name in enumerate(template.body_label)}
    joints = {name.rsplit("/", 1)[-1]: i for name, i in zip(template.joint_label, template.joint_q_start)}
    body_ids = [bodies[name] for name in body_names]
    joint_ids = [joints[name] for name in joint_names]
    corners = []
    for body_id in body_ids[-2:]:
        shapes = [
            i
            for i, body in enumerate(template.shape_body)
            if body == body_id and template.shape_flags[i] & newton.ShapeFlags.COLLIDE_SHAPES
        ]
        if len(shapes) != 1 or template.shape_type[shapes[0]] != newton.GeoType.BOX:
            raise ValueError("WBC clearance requires one collision box per foot")
        shape = shapes[0]
        points = np.array(list(product((-1, 1), repeat=3))) * np.array(template.shape_scale[shape])
        corners.append([list(wp.transform_point(template.shape_transform[shape], wp.vec3(*point))) for point in points])
    count = min(128, len(joint_pos))
    if count == 0:
        raise ValueError("WBC endpoints must not be empty")
    builder = newton.ModelBuilder()
    builder.replicate(template, count)
    model = builder.finalize(device="cpu")
    state = model.state()
    coordinates = wp.to_torch(model.joint_q).view(count, -1)
    poses = []
    for start in range(0, len(joint_pos), count):
        size = min(count, len(joint_pos) - start)
        coordinates[:size, :7] = root_pose[start : start + size]
        coordinates[:size, joint_ids] = joint_pos[start : start + size]
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        poses.append(wp.to_torch(state.body_q).view(count, template.body_count, 7)[:size, body_ids].clone())
    return torch.cat(poses), torch.tensor(corners, dtype=torch.float32)
