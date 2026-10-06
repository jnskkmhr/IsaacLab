# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Mesh-only robot targets for the Newton GL visualizer, with independent forward kinematics."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.assets import Asset, AssetBaseCfg
from isaaclab.sim import SimulationContext, SpawnerCfg
from isaaclab.sim.utils import clone, create_prim
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from pxr import Usd


@clone
def spawn_ghost_robot(
    prim_path: str,
    cfg: GhostRobotSpawnCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn the ghost's anchor without physics schemas or collision geometry.

    The asset submits only visual meshes to Newton GL when its target pose is written.
    No copy of the source articulation is added to the simulation's physics model.
    """
    return create_prim(prim_path, "Xform", translation=translation, orientation=orientation)


@configclass
class GhostRobotSpawnCfg(SpawnerCfg):
    """Source robot USD and uniform translucent material for mesh-only rendering."""

    func = spawn_ghost_robot
    usd_path: str = MISSING
    color: tuple[float, float, float] = (0.2, 0.55, 1.0)
    opacity: float = 0.25


@wp.kernel
def _transform_vertices(
    body_q: wp.array(dtype=wp.transform),
    local_points: wp.array(dtype=wp.vec3),
    local_normals: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_count: int,
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
):
    instance, vertex = wp.tid()
    transform = body_q[instance * body_count + body_ids[vertex]]
    output = instance * local_points.shape[0] + vertex
    points[output] = wp.transform_point(transform, local_points[vertex])
    normals[output] = wp.transform_vector(transform, local_normals[vertex])


class GhostRobot(Asset):
    """Render prescribed joint/root poses without a simulated articulation or contacts.

    A separate Newton model is used only for forward kinematics, never stepped by a
    solver. Source collision shapes are excluded from the rendered mesh. Instances
    correspond to the first ``max_instances`` environments; poses are world-frame XYZW.
    """

    def __init__(self, cfg: GhostRobotCfg):
        super().__init__(cfg)
        if not 0.0 <= cfg.spawn.opacity <= 1.0:
            raise ValueError("Ghost opacity must be in [0, 1].")
        if cfg.max_instances < 1:
            raise ValueError("Ghost max_instances must be positive.")
        self._model = None

    def _initialize_mesh(self, count: int, joint_names: list[str], device: str) -> None:
        import newton

        template = newton.ModelBuilder()
        template.add_usd(
            self.cfg.spawn.usd_path,
            collapse_fixed_joints=False,
            load_visual_shapes=True,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )
        if template.joint_type[0] != newton.JointType.FREE:
            raise ValueError("Ghost robot requires a floating-base source articulation.")
        coordinates = {
            name.rsplit("/", 1)[-1]: start for name, start in zip(template.joint_label, template.joint_q_start)
        }
        self._joint_ids = [coordinates[name] for name in joint_names]
        self._joint_names = tuple(joint_names)
        vertices, triangles, bodies = [], [], []
        vertex_count = 0
        for i, flags in enumerate(template.shape_flags):
            if not flags & newton.ShapeFlags.VISIBLE:
                continue
            if template.shape_type[i] != newton.GeoType.MESH:
                raise ValueError("Ghost robot currently supports mesh visual geometry only.")
            mesh = template.shape_source[i]
            transform = template.shape_transform[i]
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
            points = np.asarray(mesh.vertices) * np.asarray(template.shape_scale[i])
            points = points @ rotation.T + np.asarray(wp.transform_get_translation(transform))
            vertices.append(points)
            triangles.append(np.asarray(mesh.indices).reshape(-1, 3) + vertex_count)
            bodies.extend([template.shape_body[i]] * len(points))
            vertex_count += len(points)
        points = np.concatenate(vertices).astype(np.float32)
        indices = np.concatenate(triangles).astype(np.int32)
        normals = np.zeros_like(points)
        faces = np.cross(points[indices[:, 1]] - points[indices[:, 0]], points[indices[:, 2]] - points[indices[:, 0]])
        for corner in range(3):
            np.add.at(normals, indices[:, corner], faces)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
        builder = newton.ModelBuilder()
        builder.replicate(template, count)
        self._model = builder.finalize(device=device)
        self._state = self._model.state()
        self._count = count
        self._body_count = template.body_count
        self._coordinates = wp.to_torch(self._model.joint_q).view(count, -1)
        self._local_points = wp.array(points, dtype=wp.vec3, device=device)
        self._local_normals = wp.array(normals, dtype=wp.vec3, device=device)
        self._body_ids = wp.array(bodies, dtype=int, device=device)
        self._points = wp.empty(count * len(points), dtype=wp.vec3, device=device)
        self._normals = wp.empty_like(self._points)
        indices = indices[None] + np.arange(count)[:, None, None] * len(points)
        self._indices = wp.array(indices.astype(np.int32).flatten(), dtype=int, device=device)

    def write_pose(self, root_pose: torch.Tensor, joint_position: torch.Tensor, joint_names: list[str]) -> None:
        """Draw target poses for the configured number of instances using Newton GL.

        Args:
            root_pose: World-frame root positions and XYZW quaternions, shape (N, 7).
            joint_position: Joint angles in radians, shape (N, len(joint_names)).
            joint_names: Names defining the column order of ``joint_position``.
        """
        import newton
        from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer

        if not self.cfg.enabled or not self.cfg.spawn.visible:
            return
        viewers = [viz for viz in SimulationContext.instance().visualizers if isinstance(viz, NewtonGLVisualizer)]
        if not viewers:
            return
        count = min(len(root_pose), self.cfg.max_instances)
        if self._model is None:
            self._initialize_mesh(count, joint_names, str(joint_position.device))
        if count != self._count or tuple(joint_names) != self._joint_names:
            raise ValueError("Ghost instance count and joint ordering must stay fixed after initialization.")
        self._coordinates[:, :7] = root_pose[:count]
        self._coordinates[:, self._joint_ids] = joint_position[:count]
        newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state)
        wp.launch(
            _transform_vertices,
            dim=(count, len(self._local_points)),
            inputs=[self._state.body_q, self._local_points, self._local_normals, self._body_ids, self._body_count],
            outputs=[self._points, self._normals],
            device=self._model.device,
        )
        for viewer in viewers:
            viewer.log_mesh(
                self.cfg.prim_path + "/visual_mesh",
                points=self._points,
                indices=self._indices,
                normals=self._normals,
                color=self.cfg.spawn.color,
                opacity=self.cfg.spawn.opacity,
                backface_culling=True,
            )


@configclass
class GhostRobotCfg(AssetBaseCfg):
    """Visualization-only robot asset; no simulation state, actuators, or collision sensors."""

    class_type: type = GhostRobot
    spawn: GhostRobotSpawnCfg = MISSING
    enabled: bool = True
    max_instances: int = 20
    """Maximum number of environment targets drawn, starting with environment zero."""
