# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object geometry adapter to the current G1 compact Particle NeRD update.

Training, recorded-motion evaluation and dynamic object playback share this
adapter. Selection, blending, ballistics, displacement stabilization and tray
projection are implemented in ``inference.py``, not reimplemented here.
"""

import torch

from .inference import finish_compact_step, prepare_compact_step

SCHEMA = "object_particle_nerd_topk_v2"
PIPELINE = {
    "active_k": "auto",
    "anchor_particles": 512,
    "selection_interval": 4,
    "displacement_gate": True,
    "gate_during_training": True,
    "free_surface_ballistics": True,
    "particle_projection": "tray_floor_and_walls",
    "packing_correction": False,
    "collider_input": "pre_rigid_step",
    "support": 0.07,
}
# Pinned to g1_29dof_particle_nerd/env_cfg/physics_cfg.py, including the
# fixed-task overrides rather than the generic solver's different defaults.
CONTACT_SETTINGS = {
    "effective_mass_scale": 100.0,
    "tangent_mass_scale": 40.0,
    "horizontal_mass_scale": 8.0,
    "upward_mass_scale": 2.0,
    "normal_power": 3.0,
    "contact_iterations": 200,
    "contact_relaxation": 0.02,
    "gradient_radius": 0.065,
    "gradient_contact_band": 0.08,
    "contact_stabilization": 0.01,
    "contact_offset": 0.0,
}


def validate_checkpoint(saved):
    """Reject historical raw-displacement checkpoints for this inference method."""
    config = saved.get("config", {})
    if config.get("schema") != SCHEMA or config.get("pipeline") != PIPELINE:
        raise ValueError(
            "This task requires the retrained object Top-K checkpoint with G1 stabilization. "
            "The historical object_particle_nerd_v1 checkpoint used a different inference method."
        )


class ObjectParticleStepper:
    """Retain one object's selection cache, with the G1 update every four steps."""

    def __init__(self, kinds, sizes, friction, material, dt, *, floor_slot=1):
        if int(kinds[0]) not in (1, 2, 4) or not bool(torch.all(kinds[1:] == 1)):
            raise ValueError("Expected a box, sphere or cylinder followed by static tray boxes")
        self.material = material
        self.dt = dt
        self.geometry = {
            "kinds": kinds,
            "sizes": sizes,
            "friction": friction,
            "foot_slots": torch.tensor([0], device=kinds.device, dtype=torch.long),
            "floor_slot": floor_slot,
            "collider_kinds": kinds[:1],
            "p_radius": float(0.02 * material[11]),
            "mu_mat": float(material[3]),
            "stabilize_training": True,
        }
        self.phase = 0

    def reset(self):
        """Invalidate selection after replacing or resetting a particle history."""
        for key in list(self.geometry):
            if key.startswith("cached_") or key == "disp_buf":
                del self.geometry[key]
        self.phase = 0

    def prepare(self, history, pose, velocity):
        """Use the same selection and feature extraction during training and inference."""
        return prepare_compact_step(
            history,
            pose.contiguous(),
            velocity.contiguous(),
            self.geometry,
            self.material,
            self.dt,
            support=PIPELINE["support"],
            active_k=PIPELINE["active_k"],
            anchor_m=PIPELINE["anchor_particles"],
            reuse_selection=self.phase != 0,
        )

    def finish(self, prepared, displacement):
        """Apply the shared G1 particle update to the selected network predictions."""
        result = finish_compact_step(prepared, displacement.float())
        self.phase = (self.phase + 1) % PIPELINE["selection_interval"]
        return result

    def step(self, model, history, pose, velocity):
        prepared = self.prepare(history, pose, velocity)
        with torch.autocast(history.device.type, dtype=torch.bfloat16, enabled=history.is_cuda):
            displacement = model(*prepared.inputs)
        return history[:, -1] + self.finish(prepared, displacement)
