# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 environments on fixed sand testbeds, where rigid ground also supports the feet."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from .env_cfg.physics_cfg import RIGID_ENTRY
from .streaming_mpm_env import G1MPMEnv


@wp.kernel
def sum_rigid_foot_forces(
    contact_count: wp.array[int],
    contact_world: wp.array[int],
    contact_geom: wp.array[wp.vec2i],
    geom_to_shape: wp.array2d[int],
    shape_to_foot: wp.array[int],
    shape_is_ground: wp.array[wp.bool],
    contact_wrench: wp.array[wp.spatial_vector],
    foot_force: wp.array[wp.vec3],
):
    """Sum the world-frame forces applied to each foot across its MJWarp contacts."""
    contact = wp.tid()
    if contact < contact_count[0]:
        world = contact_world[contact]
        geoms = contact_geom[contact]
        force = wp.spatial_top(contact_wrench[contact])
        for side in range(2):
            shape = geom_to_shape[world, geoms[side]]
            other_shape = geom_to_shape[world, geoms[1 - side]]
            if shape >= 0 and other_shape >= 0 and shape_is_ground[other_shape]:
                foot = shape_to_foot[shape]
                if foot >= 0:
                    sign = float(2 * side - 1)
                    wp.atomic_add(foot_force, foot, sign * force)


class G1TestbedEnv(G1MPMEnv):
    """G1 walking on a fixed sand bed.

    The foot contact state combines the granular reaction with the MJWarp support forces of the
    floor beneath the sand and the tilting tray.
    """

    def step(self, action: torch.Tensor):
        self._rigid_contact_valid.fill_(True)
        return super().step(action)

    def reset_mpm_state(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        super().reset_mpm_state(env_ids)
        self._foot_contact[env_ids] = False
        self._foot_contact_force[env_ids] = 0.0
        self._rigid_contact_valid[env_ids] = False

    def _setup_contact_state(self) -> None:
        """Resolve the feet's MJWarp ground contacts in addition to the proxy feedback buffer."""
        self._rigid_solver = None
        super()._setup_contact_state()
        entry = self.sim.backend_manager.backend.solver._entries[RIGID_ENTRY]
        foot_bodies = self._foot_body_ids.cpu().numpy().flatten()
        body_to_foot = {int(body): i for i, body in enumerate(foot_bodies)}
        local_to_global = entry.body_local_to_global.numpy()
        shape_to_foot = np.full(entry.solver.model.shape_count, -1, dtype=np.int32)
        for shape, body in enumerate(entry.solver.model.shape_body.numpy()):
            if body >= 0:
                shape_to_foot[shape] = body_to_foot.get(int(local_to_global[body]), -1)
        self._shape_to_foot = wp.array(shape_to_foot, device=self.device)
        self._shape_is_ground = wp.array(
            [
                bool(
                    re.match(
                        r"^/World/(ground|envs/env_\d+/"
                        r"(RigidBedFloor|Sand/RigidBedFloor|Testbed/tray))(/|$)",
                        label,
                    )
                )
                for label in entry.solver.model.shape_label
            ],
            dtype=wp.bool,
            device=self.device,
        )
        capacity = entry.solver.mjw_data.contact.geom.shape[0]
        self._contact_indices = wp.array(np.arange(capacity, dtype=np.int32), device=self.device)
        self._rigid_contact_wrench = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self._rigid_foot_force = wp.zeros(len(foot_bodies), dtype=wp.vec3, device=self.device)
        self._rigid_contact_valid = torch.zeros((self.num_envs, 1, 1), dtype=torch.bool, device=self.device)
        self._rigid_solver = entry.solver
        self._refresh_contact_forces()

    def _refresh_contact_forces(self) -> None:
        """Combine MPM and rigid-ground reaction forces and update foot contact flags."""
        super()._refresh_contact_forces()
        solver = self._rigid_solver
        if solver is None:
            return
        # Add support forces from the rigid floor beneath the sand.
        mjw.contact_force(solver.mjw_model, solver.mjw_data, self._contact_indices, True, self._rigid_contact_wrench)
        self._rigid_foot_force.zero_()
        wp.launch(
            sum_rigid_foot_forces,
            dim=self._contact_indices.shape[0],
            inputs=[
                solver.mjw_data.nacon,
                solver.mjw_data.contact.worldid,
                solver.mjw_data.contact.geom,
                solver.mjc_geom_to_newton_shape,
                self._shape_to_foot,
                self._shape_is_ground,
                self._rigid_contact_wrench,
            ],
            outputs=[self._rigid_foot_force],
            device=self.device,
        )
        rigid_force = wp.to_torch(self._rigid_foot_force).reshape(self.num_envs, self.foot_count, 3)
        self._foot_contact_force += torch.where(self._rigid_contact_valid, rigid_force, 0.0)
        force_magnitude = torch.norm(self._foot_contact_force, dim=-1)
        self._foot_contact = force_magnitude > self.cfg.foot_contact_force_threshold


class G1TiltingTestbedEnv(G1TestbedEnv):
    """Drive testbed joint targets; never reposition the tray or particles during stepping."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._debug_vis_handle = None
        self._testbed_visualizer = None
        cfg.testbed_motion.validate()
        limit = math.radians(cfg.scene.testbed.spawn.joint_limit_deg)
        if any(
            abs(value) >= limit
            for bounds in (cfg.testbed_motion.roll_range, cfg.testbed_motion.pitch_range)
            for value in bounds
        ):
            raise ValueError("Testbed target ranges must lie strictly inside the mechanical joint limits.")
        super().__init__(cfg, render_mode, **kwargs)
        self._testbed_joint_ids, _ = self.scene["testbed"].find_joints(["roll", "pitch"], preserve_order=True)
        self.testbed_tray_body_id = self.scene["testbed"].find_bodies("tray")[0][0]
        self._testbed_ranges = torch.tensor(
            [cfg.testbed_motion.roll_range, cfg.testbed_motion.pitch_range], device=self.device
        )
        self._testbed_frequency = 2 * math.pi / torch.tensor(cfg.testbed_motion.periods, device=self.device)
        length, width, depth = cfg.scene.testbed.spawn.sand_bed_size
        self._testbed_marker_offset = torch.tensor(
            [length / 2 + 0.3, -width / 2 - 0.3, depth + 0.1], device=self.device
        ).expand(self.num_envs, -1)
        # Arrow assets point along local X; rotate X to the tray's local +Z normal.
        zeros = torch.zeros(self.num_envs, device=self.device)
        self._testbed_normal_arrow_rotation = quat_from_euler_xyz(zeros, zeros - math.pi / 2, zeros)
        opposite_gravity = -torch.tensor(cfg.sim.gravity, device=self.device)
        pitch = torch.atan2(-opposite_gravity[2], torch.linalg.vector_norm(opposite_gravity[:2]))
        yaw = torch.atan2(opposite_gravity[1], opposite_gravity[0])
        self._opposite_gravity_arrow_rotation = quat_from_euler_xyz(zeros, zeros + pitch, zeros + yaw)
        self._testbed_marker_indices = torch.cat((zeros, zeros + 1)).long()
        self.set_testbed_debug_vis(cfg.testbed_motion.debug_vis)

    def set_testbed_debug_vis(self, debug_vis: bool) -> None:
        """Toggle the tray-normal and opposite-gravity arrows independently of command markers."""
        if debug_vis:
            if self._testbed_visualizer is None:
                self._testbed_visualizer = VisualizationMarkers(self.cfg.testbed_motion.visualizer_cfg)
            self._testbed_visualizer.set_visibility(True)
            if self._debug_vis_handle is None:
                self._debug_vis_handle = self.sim.vis_marker_registry.add_debug_vis_callback(self)
        else:
            if self._testbed_visualizer is not None:
                self._testbed_visualizer.set_visibility(False)
            self.sim.vis_marker_registry.clear_debug_vis_callback(self)

    def _debug_vis_callback(self, event) -> None:
        """Compare the solved tray normal with opposite gravity from the same arrow origin."""
        testbed = self.scene["testbed"]
        orientation = testbed.data.body_quat_w.torch[:, self.testbed_tray_body_id]
        position = testbed.data.body_pos_w.torch[:, self.testbed_tray_body_id]
        marker_position = position + quat_apply(orientation, self._testbed_marker_offset)
        self._testbed_visualizer.visualize(
            translations=torch.cat((marker_position, marker_position)),
            orientations=torch.cat(
                (quat_mul(orientation, self._testbed_normal_arrow_rotation), self._opposite_gravity_arrow_rotation)
            ),
            marker_indices=self._testbed_marker_indices,
        )

    def close(self):
        if self._debug_vis_handle is not None:
            self.sim.vis_marker_registry.clear_debug_vis_callback(self)
        super().close()

    def step(self, action):
        motion = self.cfg.testbed_motion
        time = (self.episode_length_buf[:, None] * self.step_dt - motion.settling_time).clamp_min(0.0)
        ramp = (time / motion.ramp_time).clamp(max=1.0)
        envelope = 0.5 * (1.0 - torch.cos(math.pi * ramp))
        envelope_velocity = torch.where(
            time < motion.ramp_time, 0.5 * math.pi / motion.ramp_time * torch.sin(math.pi * ramp), 0.0
        )
        midpoint = self._testbed_ranges.mean(dim=-1)
        amplitude = 0.5 * (self._testbed_ranges[:, 1] - self._testbed_ranges[:, 0])
        phase = time * self._testbed_frequency
        wave = midpoint + amplitude * torch.sin(phase)
        target = envelope * wave
        velocity = envelope_velocity * wave + envelope * amplitude * self._testbed_frequency * torch.cos(phase)
        testbed = self.scene["testbed"]
        testbed.set_joint_position_target(target, joint_ids=self._testbed_joint_ids)
        testbed.set_joint_velocity_target(velocity, joint_ids=self._testbed_joint_ids)
        return super().step(action)

    def _reset_idx(self, env_ids):
        # Discontinuous resets are confined to episode boundaries, together with the robot and sand.
        testbed = self.scene["testbed"]
        positions = testbed.data.default_joint_pos.torch[env_ids].clone()
        velocities = torch.zeros_like(positions)
        testbed.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)
        testbed.set_joint_position_target(positions, env_ids=env_ids)
        testbed.set_joint_velocity_target(velocities, env_ids=env_ids)
        super()._reset_idx(env_ids)
