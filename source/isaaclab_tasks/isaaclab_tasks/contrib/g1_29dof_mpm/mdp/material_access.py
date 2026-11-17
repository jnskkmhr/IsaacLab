# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Port of the source task material accessors, bound to this environment backend."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import warp as wp
from isaaclab_newton.physics.mpm_solver import implicit_mpm_solvers
from newton import Model, ModelFlags

from isaaclab.utils.warp import ProxyArray


class MPMMaterialAccess:
    """Bind the existing material getters/setters to one Newton manager."""

    def __init__(self, manager):
        self.manager = manager

    _material_parameter_names = (
        "density",
        "young_modulus",
        "poisson_ratio",
        "viscosity",
        "friction",
        "damping",
        "yield_pressure",
        "tensile_yield_ratio",
        "yield_stress",
        "hardening",
        "dilatancy",
        "hardening_rate",
        "softening_rate",
    )

    def get_particle_material_parameters(
        self,
        particle_ids: Sequence[int] | wp.array | slice | None = None,
        parameters: Sequence[str] | None = None,
    ) -> dict[str, ProxyArray]:
        """Read independent material arrays on the simulation device.

        Args:
            particle_ids: Global Newton particle indices, as a sequence, slice, or one-dimensional
                Warp int32/int64 array on the simulation device. None selects all particles.
            parameters: Names from :class:`MPMParticleMaterialCfg`. None selects all material parameters.

        Returns:
            One ProxyArray snapshot per parameter, in the requested particle order. Use ``.warp``
            for Warp operations or ``.torch`` for Torch operations. Density [kg/m^3] is mass divided
            by represented volume, ``8 * radius**3``; zero-mass particles report zero density.

        Raises:
            ValueError: A parameter name or particle selector is invalid.
            IndexError: A particle index is outside the model.
            RuntimeError: No implicit MPM solver is initialized.
        """
        parameter_names = self._material_parameter_names if parameters is None else parameters
        model, particle_indices = self._material_model_and_selection(particle_ids, parameter_names)
        invalid_indices = wp.zeros(1, dtype=wp.int32, device=model.device)
        material_parameters = {}
        for parameter_name in parameter_names:
            is_density = parameter_name == "density"
            source = model.particle_mass if is_density else getattr(model.mpm, parameter_name)
            parameter_values = wp.empty(particle_indices.shape[0], dtype=wp.float32, device=model.device)
            wp.launch(
                _read_particle_material,
                dim=particle_indices.shape[0],
                inputs=[source, model.particle_radius, particle_indices, is_density],
                outputs=[parameter_values, invalid_indices],
                device=model.device,
            )
            material_parameters[parameter_name] = ProxyArray(parameter_values)
        if invalid_indices.numpy()[0]:
            raise IndexError("particle_ids contains indices outside the Newton model's particle arrays.")
        return material_parameters

    def set_particle_material_parameters(
        self,
        material_parameters: dict[str, float | wp.array],
        particle_ids: Sequence[int] | wp.array | slice | None = None,
    ) -> None:
        """Update material arrays and queue solver synchronization before the next physics step.

        Args:
            material_parameters: Parameter names mapped to scalars or Warp float32 arrays shaped
                ``(num_selected_particles,)`` on the simulation device. Density [kg/m^3] updates
                mass and inverse mass at fixed particle radius; zero-mass particles remain kinematic.
                Damping is a relaxation time [s], as in the cfg. Pass ProxyArray inputs via ``.warp``.
            particle_ids: Global Newton particle indices, as a sequence, slice, or one-dimensional
                Warp int32/int64 array on the simulation device. None selects all particles.

        This updates the shared Newton model. Coupled solvers must support particle-property
        synchronization through ``MODEL_PROPERTIES``. Particle deformation history is preserved.
        All parameters are validated before any model array is written.

        Raises:
            ValueError: A parameter name, array shape, device, dtype, or material value is invalid.
            IndexError: A particle index is outside the model.
            RuntimeError: No implicit MPM solver is initialized.
        """
        if not material_parameters:
            return
        model, particle_indices = self._material_model_and_selection(particle_ids, material_parameters)
        material_parameters = self._prepare_material_parameters(model, particle_indices, material_parameters)
        if particle_indices.shape[0] == 0:
            return
        for parameter_name, parameter_values in material_parameters.items():
            is_density = parameter_name == "density"
            destination = model.particle_mass if is_density else getattr(model.mpm, parameter_name)
            wp.launch(
                _write_particle_material,
                dim=particle_indices.shape[0],
                inputs=[parameter_values, particle_indices, model.particle_radius, is_density],
                outputs=[destination, model.particle_inv_mass],
                device=model.device,
            )
        self.manager.add_model_change(ModelFlags.MODEL_PROPERTIES)

    def _material_model_and_selection(
        self, particle_ids: Sequence[int] | wp.array | slice | None, parameter_names: Iterable[str]
    ) -> tuple[Model, wp.array]:
        """Resolve the shared Newton model and particle indices for both material accessors."""
        if self.manager.backend is None or not implicit_mpm_solvers(self.manager.backend.solver):
            raise RuntimeError("An implicit MPM solver must be initialized before accessing particle materials.")
        unknown_parameters = set(parameter_names) - set(self._material_parameter_names)
        if unknown_parameters:
            raise ValueError(f"Unknown MPM material parameters: {sorted(unknown_parameters)}")
        model = self.manager.backend.model
        if particle_ids is None:
            particle_ids = slice(None)
        if isinstance(particle_ids, slice):
            if particle_ids.step is not None and particle_ids.step <= 0:
                raise ValueError("Particle slices must have a positive step.")
            particle_ids = range(model.particle_count)[particle_ids]
        if not isinstance(particle_ids, wp.array):
            particle_ids = wp.array(particle_ids, dtype=wp.int32, device=model.device)
        if (
            particle_ids.ndim != 1
            or particle_ids.dtype not in (wp.int32, wp.int64)
            or particle_ids.device != model.device
        ):
            raise ValueError("particle_ids must be a one-dimensional Warp int32/int64 array on the simulation device.")
        return model, particle_ids

    def _prepare_material_parameters(
        self, model: Model, particle_indices: wp.array, material_parameters: dict[str, float | wp.array]
    ) -> dict[str, wp.array]:
        """Normalize inputs and validate the whole batch before writing, with one device readback."""
        prepared_parameters = {}
        invalid_parameters = wp.zeros(len(material_parameters), dtype=wp.int32, device=model.device)
        for parameter_index, (parameter_name, parameter_values) in enumerate(material_parameters.items()):
            if isinstance(parameter_values, (float, int)):
                parameter_values = wp.full(1, float(parameter_values), dtype=wp.float32, device=model.device)
            elif (
                not isinstance(parameter_values, wp.array)
                or parameter_values.dtype != wp.float32
                or parameter_values.device != model.device
                or parameter_values.shape != particle_indices.shape
            ):
                raise ValueError(
                    f"{parameter_name} must be a scalar or Warp float32 array with shape"
                    f" {particle_indices.shape} on {model.device}."
                )
            lower_bound = -1.0 if parameter_name == "poisson_ratio" else 0.0
            upper_bound = {"poisson_ratio": 0.5, "tensile_yield_ratio": 1.0}.get(parameter_name, float("inf"))
            wp.launch(
                _validate_particle_material,
                dim=particle_indices.shape[0],
                inputs=[
                    parameter_values,
                    particle_indices,
                    model.particle_radius,
                    lower_bound,
                    upper_bound,
                    parameter_name in ("density", "young_modulus", "poisson_ratio"),
                    parameter_name == "poisson_ratio",
                    parameter_name == "density",
                    parameter_index,
                ],
                outputs=[invalid_parameters],
                device=model.device,
            )
            prepared_parameters[parameter_name] = parameter_values
        for parameter_name, error in zip(prepared_parameters, invalid_parameters.numpy(), strict=True):
            if error == 2:
                raise IndexError("particle_ids contains indices outside the Newton model's particle arrays.")
            if error:
                raise ValueError(f"Invalid MPM material value for {parameter_name}.")
        return prepared_parameters


@wp.kernel
def _read_particle_material(
    source: wp.array[float],
    particle_radius: wp.array[float],
    particle_indices: wp.array(dtype=Any),
    is_density: bool,
    material_values: wp.array[float],
    invalid_indices: wp.array[int],
):
    """Gather one material parameter in the requested particle order.

    Launch one thread per selected particle. Density is computed from particle mass
    and represented volume, ``8 * radius**3``. The source arrays are not modified.

    Args:
        source: Model-wide parameter values, or particle masses [kg] when reading density.
        particle_radius: Model-wide particle radii [m].
        particle_indices: Global particle indices in the requested output order.
        is_density: Whether to convert particle mass to density [kg/m^3].
        material_values: Output values, one per selected particle.
        invalid_indices: Zero-initialized, single-element status array. Set to 1 if any
            index is out of bounds; the corresponding output value is left unwritten.
    """
    selection_index = wp.tid()
    particle_index = particle_indices[selection_index]
    if particle_index < 0 or particle_index >= source.shape[0]:
        wp.atomic_max(invalid_indices, 0, 1)
        return
    material_value = source[particle_index]
    if is_density:
        radius = particle_radius[particle_index]
        material_value /= 8.0 * radius * radius * radius
    material_values[selection_index] = material_value


@wp.kernel
def _validate_particle_material(
    material_values: wp.array[float],
    particle_indices: wp.array(dtype=Any),
    particle_radius: wp.array[float],
    lower_bound: float,
    upper_bound: float,
    exclude_lower_bound: bool,
    exclude_upper_bound: bool,
    is_density: bool,
    parameter_index: int,
    invalid_parameters: wp.array[int],
):
    """Validate one parameter of a material-update batch without modifying the model.

    Launch one thread per selected particle. Values must be finite and within the
    specified bounds. Density must also produce a finite, positive particle mass.

    Args:
        material_values: One value to broadcast, or one value per selected particle.
        particle_indices: Global particle indices to validate.
        particle_radius: Model-wide particle radii [m], used to validate density-derived mass.
        lower_bound: Minimum allowed value, in the parameter's units.
        upper_bound: Maximum allowed value, in the parameter's units.
        exclude_lower_bound: Whether equality with the lower bound is invalid.
        exclude_upper_bound: Whether equality with the upper bound is invalid.
        is_density: Whether values represent density [kg/m^3].
        parameter_index: Slot for this parameter in the batch status array.
        invalid_parameters: Zero-initialized status array, one element per parameter.
            Records 1 for invalid values or 2 for invalid particle indices, with 2 taking priority.
    """
    selection_index = wp.tid()
    particle_index = particle_indices[selection_index]
    if particle_index < 0 or particle_index >= particle_radius.shape[0]:
        wp.atomic_max(invalid_parameters, parameter_index, 2)
        return
    value_index = wp.where(material_values.shape[0] == 1, 0, selection_index)
    material_value = material_values[value_index]
    valid = wp.isfinite(material_value) and material_value >= lower_bound and material_value <= upper_bound
    if exclude_lower_bound:
        valid = valid and material_value > lower_bound
    if exclude_upper_bound:
        valid = valid and material_value < upper_bound
    if is_density:
        radius = particle_radius[particle_index]
        particle_mass = material_value * (8.0 * radius * radius * radius)
        valid = valid and wp.isfinite(particle_mass) and particle_mass > 0.0
    if not valid:
        wp.atomic_max(invalid_parameters, parameter_index, 1)


@wp.kernel
def _write_particle_material(
    material_values: wp.array[float],
    particle_indices: wp.array(dtype=Any),
    particle_radius: wp.array[float],
    is_density: bool,
    destination: wp.array[float],
    particle_inv_mass: wp.array[float],
):
    """Scatter validated material values into the shared Newton model.

    Launch one thread per selected particle after the entire batch passes validation.
    Density writes update mass and inverse mass while preserving zero-mass particles.

    Args:
        material_values: One value to broadcast, or one value per selected particle.
        particle_indices: Validated global particle indices to update.
        particle_radius: Model-wide particle radii [m].
        is_density: Whether values represent density [kg/m^3] rather than a direct parameter.
        destination: Model-wide parameter array to update, or particle masses [kg] for
            density writes. Existing mass determines whether each particle remains kinematic.
        particle_inv_mass: Model-wide inverse masses [1/kg], updated only for density writes.
            Zero-mass particles retain zero mass and receive zero inverse mass.
    """
    selection_index = wp.tid()
    particle_index = particle_indices[selection_index]
    value_index = wp.where(material_values.shape[0] == 1, 0, selection_index)
    material_value = material_values[value_index]
    if is_density:
        # Zero simulation mass identifies kinematic particles; preserve that classification.
        if destination[particle_index] > 0.0:
            radius = particle_radius[particle_index]
            particle_mass = material_value * (8.0 * radius * radius * radius)
            destination[particle_index] = particle_mass
            particle_inv_mass[particle_index] = 1.0 / particle_mass
        else:
            particle_inv_mass[particle_index] = 0.0
    else:
        destination[particle_index] = material_value
