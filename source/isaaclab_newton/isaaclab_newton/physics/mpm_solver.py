# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit MPM solver implementation and reset helpers."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields
from typing import TYPE_CHECKING

import warp as wp
from newton import BodyFlags, Contacts, Control, GeoType, Model, ModelBuilder, State, StateFlags
from newton.solvers import SolverBase, SolverImplicitMPM
from warp.fem import TemporaryStore

from isaaclab.sim import SimulationContext

from .mpm_solver_cfg import MPMSleepingCfg, MPMSolverCfg, MPMWorkingSetCfg, StrainActivationCfg, StreamingCfg
from .newton_solver import NewtonSolver

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim.spawners.mpm import StreamingMPMGridCfg
    from .newton_backend import NewtonBackend
    from .newton_manager import NewtonManager


_deferred_status_checks: ContextVar[tuple[SolverImplicitMPM, ...]] = ContextVar(
    "mpm_deferred_status_checks", default=()
)


def _canonical_collider_velocity_mode(solver_cfg: MPMSolverCfg) -> str:
    """Resolve deprecated collider-velocity aliases used by older configurations."""
    collider_velocity_mode = solver_cfg.collider_velocity_mode
    deprecated_velocity_modes = {
        "instantaneous": "forward",
        "finite_difference": "backward",
    }
    if replacement := deprecated_velocity_modes.get(collider_velocity_mode):
        warnings.warn(
            f"collider_velocity_mode={collider_velocity_mode!r} is deprecated; use {replacement!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        collider_velocity_mode = replacement
    return collider_velocity_mode


def _make_solver_config(solver_cfg: MPMSolverCfg, scene_prim: Usd.Prim | None = None) -> SolverImplicitMPM.Config:
    """Build Newton's implicit MPM config, consuming authored USD when available."""
    collider_velocity_mode = _canonical_collider_velocity_mode(solver_cfg)
    values = {
        "max_iterations": solver_cfg.max_iterations,
        "tolerance": solver_cfg.tolerance,
        "solver": solver_cfg.solver,
        "warmstart_mode": solver_cfg.warmstart_mode,
        "collider_velocity_mode": collider_velocity_mode,
        "voxel_size": solver_cfg.voxel_size,
        "grid_type": solver_cfg.grid_type,
        "grid_padding": solver_cfg.grid_padding,
        "max_active_cell_count": solver_cfg.max_active_cell_count,
        "max_leaf_node_count": solver_cfg.max_leaf_node_count,
        "max_lower_node_count": solver_cfg.max_lower_node_count,
        "max_upper_node_count": solver_cfg.max_upper_node_count,
        "separate_worlds": solver_cfg.separate_worlds,
        "transfer_scheme": solver_cfg.transfer_scheme,
        "integration_scheme": solver_cfg.integration_scheme,
        "critical_fraction": solver_cfg.critical_fraction,
        "air_drag": solver_cfg.air_drag,
        "collider_normal_from_sdf_gradient": solver_cfg.collider_normal_from_sdf_gradient,
        "collider_basis": solver_cfg.collider_basis,
        "strain_basis": solver_cfg.strain_basis,
        "velocity_basis": solver_cfg.velocity_basis,
    }
    if "reuse_grid_topology" in {field.name for field in fields(SolverImplicitMPM.Config)}:
        values["reuse_grid_topology"] = solver_cfg.reuse_grid_topology
    elif solver_cfg.reuse_grid_topology:
        raise RuntimeError("Grid topology reuse requires a compatible Newton build providing reuse_grid_topology.")
    if scene_prim is None:
        return SolverImplicitMPM.Config(**values)

    config = SolverImplicitMPM.Config.create_from_usd(scene_prim)
    # Retain the cfg's Python values after validating the authored schema. USD
    # float attributes otherwise quantize grid-sensitive values to float32.
    for name, value in values.items():
        setattr(config, name, value)
    return config


def _author_mpm_scene_config(scene_prim: Usd.Prim, solver_cfg: MPMSolverCfg) -> None:
    """Author schema-representable MPM solver settings on a physics scene."""
    from pxr import Sdf, Vt  # noqa: PLC0415

    # Preserve the raw API metadata when Kit's registry predates the installed codeless schemas.
    if not scene_prim.AddAppliedSchema("NewtonMPMSceneAPI"):
        raise RuntimeError(f"Failed to apply NewtonMPMSceneAPI to '{scene_prim.GetPath()}'.")

    rheology_solvers = (solver_cfg.solver,) if isinstance(solver_cfg.solver, str) else solver_cfg.solver

    attributes = {
        "newton:maxSolverIterations": solver_cfg.max_iterations,
        "newton:mpm:tolerance": solver_cfg.tolerance,
        "newton:mpm:rheologySolvers": Vt.TokenArray(rheology_solvers),
        "newton:mpm:voxelSize": solver_cfg.voxel_size,
        "newton:mpm:gridType": solver_cfg.grid_type,
        "newton:mpm:gridPadding": solver_cfg.grid_padding,
        "newton:mpm:maxActiveCellCount": solver_cfg.max_active_cell_count,
        "newton:mpm:transferScheme": solver_cfg.transfer_scheme,
        "newton:mpm:integrationScheme": solver_cfg.integration_scheme,
        "newton:mpm:criticalFraction": solver_cfg.critical_fraction,
        "newton:mpm:airDrag": solver_cfg.air_drag,
    }
    attributes.update(_basis_attributes("collider", solver_cfg.collider_basis))
    attributes.update(_basis_attributes("strain", solver_cfg.strain_basis))
    attributes.update(_basis_attributes("velocity", solver_cfg.velocity_basis))
    for name, value in attributes.items():
        if name == "newton:mpm:rheologySolvers":
            type_name = Sdf.ValueTypeNames.TokenArray
        elif isinstance(value, bool):
            type_name = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            type_name = Sdf.ValueTypeNames.Int
        elif isinstance(value, str):
            type_name = Sdf.ValueTypeNames.Token
        else:
            type_name = Sdf.ValueTypeNames.Float
        scene_prim.CreateAttribute(name, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(value)


def _basis_attributes(prefix: str, basis: str) -> dict[str, object]:
    """Expand a compact Newton basis name into newton-usd-schemas attributes."""
    if basis.startswith("pic"):
        if prefix == "velocity":
            raise ValueError("The MPM velocity basis cannot use a particle basis.")
        basis_type, order, discontinuous = "particle", 0, False
    else:
        matched = re.fullmatch(r"([PQBS])([0-9]+)(d?)", basis)
        if matched is None:
            raise ValueError(f"Unsupported MPM {prefix} basis {basis!r}.")
        basis_prefix = matched.group(1)
        order = int(matched.group(2))
        discontinuous = order == 0 or bool(matched.group(3))
        if basis_prefix == "P" and order > 0 and not discontinuous:
            raise ValueError(f"Unsupported MPM {prefix} basis {basis!r}: positive-order P bases must be discontinuous.")
        if (prefix == "strain" and basis_prefix in "BS") or (
            prefix == "velocity" and (basis_prefix not in "QB" or not 1 <= order <= 3)
        ):
            return {}
        if basis_prefix in "BS" and not 1 <= order <= 3:
            return {}
        basis_type = {
            "P": "linear",
            "Q": "trilinear",
            "B": "bspline",
            "S": "serendipity",
        }[basis_prefix]

    namespace = f"newton:mpm:{prefix}"
    attributes: dict[str, object] = {
        f"{namespace}BasisType": basis_type,
        f"{namespace}BasisOrder": order,
    }
    if prefix != "velocity":
        attributes[f"{namespace}DiscontinuousBasis"] = discontinuous
    return attributes


def implicit_mpm_solvers(solver: SolverBase | None) -> tuple[SolverImplicitMPM, ...]:
    """Return a direct implicit-MPM solver, or the implicit-MPM entries of a coupled solver."""
    if isinstance(solver, SolverImplicitMPM):
        return (solver,)
    if solver is None or not hasattr(solver, "entry_names") or not hasattr(solver, "solver"):
        return ()
    return tuple(
        entry_solver
        for name in solver.entry_names()
        if isinstance((entry_solver := solver.solver(name)), SolverImplicitMPM)
    )


def check_mpm_reset_status(solver: SolverBase) -> None:
    """Validate a deferred window before a Lab reset can clear its native fault status.

    Outside deferred collection this is a no-op. Reset boundaries can synchronize; ordinary replay does not call it.
    """
    deferred = _deferred_status_checks.get()
    if deferred:
        for selected in implicit_mpm_solvers(solver):
            if selected in deferred:
                selected.check_sparse_grid_rebuild_status()


def _runtime_mpm_solver(manager: NewtonManager, solver: SolverImplicitMPM | None) -> SolverImplicitMPM:
    """Select an owned solver without guessing a coupled entry or particle ordering."""
    if manager.backend is None:
        raise RuntimeError("Initialize the Newton backend before configuring MPM runtime settings.")
    selected = manager.backend.solver if solver is None else solver
    if selected not in implicit_mpm_solvers(manager.backend.solver):
        raise ValueError(
            "Select an initialized implicit MPM solver owned by this manager; specify coupled entries explicitly."
        )
    if selected.model.device.is_capturing:
        raise RuntimeError("Configure MPM runtime settings outside CUDA graph capture.")
    return selected


def _world_body_indices(model: Model, name: str) -> list[int]:
    """Return the one body with this leaf name in each world, ordered by world."""
    body_worlds = model.body_world.numpy()
    indices = []
    for world in range(model.world_count):
        matches = [
            index
            for index, label in enumerate(model.body_label)
            if body_worlds[index] == world and label and label.rsplit("/", 1)[-1] == name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {name!r} body in environment {world}, got {matches}.")
        indices.append(matches[0])
    return indices


def configure_mpm_sleeping(
    manager: NewtonManager, cfg: MPMSleepingCfg | None, *, solver: SolverImplicitMPM | None = None
) -> None:
    """Apply native activity settings without advancing or resetting the scene.

    Settle the scene first, then step its controller to classify activity before installing a bounded working set.
    ``None`` does nothing. Existing status is checked and the step graph invalidated before configuration.
    Native Newton validates supported devices, grids, and thresholds. These are setup operations, not per-step work.

    A region follows the configured bodies through a pose buffer that a graph-safe hook refreshes every step.

    Args:
        manager: Manager owning the active Newton backend.
        cfg: Activity settings, or None to leave the solver unchanged.
        solver: Native implicit MPM solver. Omit for a standalone MPM backend; select coupled entries explicitly.

    Raises:
        RuntimeError: The native feature is missing, capture is active, or the previous step failed.
        ValueError: Neither a planner nor a region is set, a region body is missing or ambiguous, the solver is
            not owned by this manager, or Newton rejects the configuration.
    """
    if cfg is None:
        return
    selected = _runtime_mpm_solver(manager, solver)
    if not hasattr(selected, "set_particle_sleeping"):
        raise RuntimeError("Particle sleeping requires a compatible Newton build providing set_particle_sleeping().")
    cfg.validate()
    if cfg.enabled and cfg.strain is None and cfg.region is None:
        raise ValueError("Enabled MPM sleeping needs strain thresholds, a region, or both.")
    selected.check_sparse_grid_rebuild_status()
    backend = manager.backend
    backend.invalidate_graph()
    strain = StrainActivationCfg() if cfg.strain is None else cfg.strain
    selected.set_particle_sleeping(enabled=cfg.enabled, **strain.to_dict())
    region = cfg.region
    if not cfg.enabled or region is None:
        return
    if not hasattr(selected, "set_particle_active_regions"):
        raise RuntimeError("Activity regions require a Newton build providing set_particle_active_regions().")
    model = backend.model
    names = [region.body_names] if isinstance(region.body_names, str) else list(region.body_names)
    body_indices = [index for name in names for index in _world_body_indices(model, name)]
    # The solver states swap every step, so the region reads a persistent copy of the body poses.
    body_q = wp.clone(backend.state_0.body_q)
    selected.set_particle_active_regions(
        body_indices=body_indices,
        half_extents=[0.5 * extent for extent in region.size],
        offsets=region.offset,
        body_q=body_q,
        body_world=model.body_world,
        keep_material_motion=region.keep_material_motion,
        awake_inside=cfg.strain is None,
    )
    manager.register_post_actuator_callback(lambda: wp.copy(body_q, manager.backend.state_0.body_q))


def configure_mpm_working_set(
    manager: NewtonManager,
    cfg: MPMWorkingSetCfg | None,
    *,
    solver: SolverImplicitMPM | None = None,
    state: State | None = None,
) -> None:
    """Install bounded native evaluation buffers after settling and activity classification.

    ``None`` does nothing. Existing status is checked and the step graph invalidated before changing buffer layout.
    The next manager step warms and captures the new layout. All backing particle states stay on GPU. Initial settling
    and resets can wake more particles than the pool holds; choose a compatible preparation and reset policy.

    Args:
        manager: Manager owning the active Newton backend.
        cfg: Buffer capacities and optional page size, or None to leave the solver unchanged.
        solver: Native implicit MPM solver. Omit for a standalone MPM backend.
        state: Current solver input state. Required for a coupled entry; a standalone backend uses its state_0.

    Raises:
        RuntimeError: The feature or current state is missing, capture is active, or the previous step failed.
        ValueError: A coupled input state is omitted, the solver is not owned, or Newton rejects the configuration.
    """
    if cfg is None:
        return
    selected = _runtime_mpm_solver(manager, solver)
    if not hasattr(selected, "set_particle_working_set"):
        raise RuntimeError(
            "Bounded MPM evaluation requires a compatible Newton build providing set_particle_working_set()."
        )
    if state is None:
        if selected is not manager.backend.solver:
            raise ValueError("Provide the coupled MPM entry's current input state explicitly.")
        state = manager.backend.state_0
    if state is None:
        raise RuntimeError("The current MPM input state is not initialized.")
    cfg.validate()
    selected.check_sparse_grid_rebuild_status()
    manager.backend.invalidate_graph()
    selected.set_particle_working_set(state, **cfg.to_dict())


def configure_mpm_streaming(
    manager: NewtonManager,
    cfg: StreamingCfg | None,
    *,
    surface_mesh: wp.Mesh,
    world_origins: Sequence[Sequence[float]],
    spawn_heights: Sequence[float],
    solver: SolverImplicitMPM | None = None,
    state: State | None = None,
    particle_spawn_cfg: StreamingMPMGridCfg | None = None,
    dt: float | None = None,
):
    """Make the spawned particles a pool that follows the configured anchor bodies.

    ``None`` does nothing. The pool is recycled before every solver substep and its kinematic ring is
    pinned after every step; both hooks are graph-safe and stay registered with the backend. Existing
    status is checked and the step graph invalidated before configuration.

    Args:
        manager: Manager owning the active Newton backend.
        cfg: Pool settings, or None to leave every particle in place.
        surface_mesh: Surface on which recycled particles are placed.
        world_origins: Per-environment origins [m], shape ``(num_envs, 3)``; their XY phases the tile grid.
        spawn_heights: Per-environment Z [m] of the surface the particles were spawned on.
        solver: Native implicit MPM solver. Omit for a standalone MPM backend; select coupled entries explicitly.
        state: Current solver input state. Required for a coupled entry; a standalone backend uses its state_0.
        particle_spawn_cfg: Spawner that created the pool. Required with a tile cache, which samples the
            template of newly generated tiles from it.
        dt: Physics step [s]. Required with a tile cache, which settles its tile template at this step.

    Returns:
        Newton's streaming pool, or None when ``cfg`` is None.

    Raises:
        RuntimeError: The native feature or current state is missing, capture is active, or the previous step failed.
        ValueError: An anchor body is missing or ambiguous, a tile cache lacks its spawner or step, or Newton
            rejects the configuration.
    """
    if cfg is None:
        return None
    selected = _runtime_mpm_solver(manager, solver)
    if not hasattr(selected, "set_particle_streaming"):
        raise RuntimeError("Particle streaming requires a compatible Newton build providing set_particle_streaming().")
    backend = manager.backend
    if state is None:
        if selected is not backend.solver:
            raise ValueError("Provide the coupled MPM entry's current input state explicitly.")
        state = backend.state_0
    if state is None:
        raise RuntimeError("The current MPM input state is not initialized.")
    cfg.validate()
    names = [cfg.body_names] if isinstance(cfg.body_names, str) else list(cfg.body_names)
    model = backend.model
    cache = None
    if cfg.cache is None:
        if len(names) != 1:
            raise ValueError("A streaming pool without a tile cache follows exactly one anchor body.")
        body_indices = _world_body_indices(model, names[0])
    else:
        if particle_spawn_cfg is None or dt is None:
            raise ValueError("A tile cache needs the pool's spawner configuration and the physics step.")
        # One row per environment, one column per anchor.
        body_indices = list(zip(*(_world_body_indices(model, name) for name in names), strict=True))
        # Imported here so that loading the physics configuration does not load USD.
        from ..sim.spawners.mpm import sample_particle_tile  # noqa: PLC0415

        template, _, radius = sample_particle_tile(particle_spawn_cfg)
        cache = {
            "depth": particle_spawn_cfg.upper[2] - particle_spawn_cfg.lower[2],
            "tile_template": template,
            "tile_radius": radius,
            "max_cached_particles": cfg.cache.max_cached_particles_per_env,
            "max_visited_tiles": cfg.cache.max_visited_tiles_per_env,
            "settle_time": cfg.cache.settle_time_s,
            "settle_speed": cfg.cache.settle_speed,
            "settle_dt": dt,
        }
    selected.check_sparse_grid_rebuild_status()
    backend.invalidate_graph()
    streaming = selected.set_particle_streaming(
        state,
        body_indices=body_indices,
        size=cfg.size,
        margin=cfg.margin,
        tile_size=cfg.tile_size,
        surface_mesh=surface_mesh,
        world_origins=world_origins,
        spawn_heights=spawn_heights,
        model=model,
        cache=cache,
    )
    # Place the initial pools before the first step, then recycle inside the captured step.
    streaming.update(backend.state_0)
    streaming.check_status()
    manager.register_state_force_callback(streaming.update)
    manager.register_post_step_callback(lambda: streaming.post_step(manager.backend.state_0))
    return streaming


@contextmanager
def defer_mpm_status_checks(manager: NewtonManager, *, solvers: Sequence[SolverImplicitMPM] | None = None):
    """Collect a guarded MPM rollout without a per-step status readback, then validate it.

    The default per-step fail-fast policy is unchanged outside this scope. Debug replay still checks every step.
    Each successful scope exit checks all selected solvers; an overflow raises and invalidates the entire window.
    Consume collected transitions or update a policy only after successful exit. A body exception also rejects the
    window and restores ordinary checking. Nested scopes validate independently.

    Use Lab's reset interfaces inside the scope; direct native resets are unsupported because they clear fault status.
    Lab resets check before clearing status and can synchronize even inside the scope. Prepare the scene before entry.
    Status validation does not replace task-specific finite-state or physical checks.
    This requires native guarded CUDA MPM with a persistent ``step_status``. The device guard protects particle
    buffers, but coupled rigid bodies may continue after an MPM fault. This scope does not provide GPU validity masks
    for a learner or make an unbounded reset compatible with a compact pool.

    Args:
        manager: Manager owning the active Newton backend.
        solvers: Owned implicit MPM solvers to validate. None selects every MPM entry.

    Raises:
        RuntimeError: No guarded MPM solver is available, capture is active, or final validation fails.
        ValueError: A selected solver is not owned by this manager.
    """
    root_solver = getattr(manager.backend, "solver", None)
    selected = implicit_mpm_solvers(root_solver) if solvers is None else tuple(dict.fromkeys(solvers))
    if not selected:
        raise RuntimeError("Deferred validation requires an initialized guarded MPM solver.")
    for solver in selected:
        _runtime_mpm_solver(manager, solver)
        if getattr(solver, "step_status", None) is None:
            raise RuntimeError("Deferred validation requires native guarded CUDA MPM with persistent step_status.")
        solver.check_sparse_grid_rebuild_status()
    token = _deferred_status_checks.set(_deferred_status_checks.get() + selected)
    try:
        yield
        if getattr(manager.backend, "solver", None) is not root_solver:
            raise RuntimeError("The Newton backend changed during deferred collection; discard the rollout.")
        for solver in selected:
            solver.check_sparse_grid_rebuild_status()
    finally:
        _deferred_status_checks.reset(token)


def mpm_supports_graph_capture(solver: SolverImplicitMPM) -> bool:
    """Return whether an implicit-MPM solver satisfies Newton's capture contract."""
    if solver.grid_type == "fixed":
        # An unbounded active partition reads its cell count back to the CPU on every step.
        return solver.max_active_cell_count > 0
    if solver.grid_type != "sparse":
        return False
    strain_rebuild_safe = solver.strain_basis.startswith("pic") or solver.strain_basis in ("P0", "P1d", "Q1d", "Q1")
    collider_rebuild_safe = solver.collider_basis.startswith("pic") or solver.collider_basis in ("Q1", "S2", "S3")
    return (
        solver.max_active_cell_count > 0
        and solver.grid_padding == 0
        and solver.velocity_basis in ("Q1", "B2")
        and strain_rebuild_safe
        and collider_rebuild_safe
    )


def mpm_solver_config(cfg: MPMSolverCfg) -> SolverImplicitMPM.Config:
    """Return Newton's implicit MPM configuration, authored on the active physics scene first."""
    scene_prim = None
    sim = SimulationContext.instance()
    if sim is not None:
        scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        _author_mpm_scene_config(scene_prim, cfg)
    return _make_solver_config(cfg, scene_prim)


class MPMSolver(NewtonSolver):
    """Newton's implicit MPM solver, which steps particle materials in place on one state."""

    cfg: MPMSolverCfg

    solver_class = SolverImplicitMPM
    single_state = True
    contact_sensors = False

    @classmethod
    def prepare_builder(cls, cfg: MPMSolverCfg, builder: ModelBuilder) -> None:
        """Normalize rigid colliders before solver construction.

        Newton's implicit MPM treats positive-mass body colliders as finite-mass colliders, so kinematic bodies lose
        their mass and inertia, matching Newton's MPM examples. The solver accepts only the triangle-mesh geometry
        type, so convex meshes are classified as meshes without changing their geometry.
        """
        kinematic_flag = int(BodyFlags.KINEMATIC)
        for body_id, flags in enumerate(builder.body_flags):
            if int(flags) & kinematic_flag:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inertia[body_id] = wp.mat33()
                builder.body_inv_inertia[body_id] = wp.mat33()
        for shape_id, shape_type in enumerate(builder.shape_type):
            if shape_type == GeoType.CONVEX_MESH:
                builder.shape_type[shape_id] = GeoType.MESH

    @classmethod
    def coupling_error(cls, cfg: MPMSolverCfg, in_place: bool) -> str | None:
        """MPM entries step in place, and the per-substep collider projection runs outside Newton's coupled step."""
        if cfg.project_outside_colliders:
            return (
                "enables MPMSolverCfg.project_outside_colliders, whose per-substep projection cannot yet run inside a"
                " coupled-solver entry"
            )
        if not in_place:
            return "uses MPMSolverCfg and must set in_place=True"
        return None

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverImplicitMPM:
        """Construct implicit MPM from the configuration authored on the active physics scene."""
        return self.solver_class(model, mpm_solver_config(self.cfg), temporary_store=TemporaryStore())

    def applies_body_forces(self) -> bool:
        return False

    def supports_capture(self) -> bool:
        return mpm_supports_graph_capture(self.native)

    def uses_collision_pipeline(self) -> bool:
        return False

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Run one implicit MPM substep, optionally projecting particles out of collider interiors."""
        self.native.step(state_in, state_out, control, contacts, dt)
        if self.cfg.project_outside_colliders:
            self.native.project_outside(state_out, state_out, dt)

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Keep implicit-MPM history across automatic asset resets.

        Shared-grid MPM cannot reset history for a subset of worlds. Tasks with independent MPM worlds that need an
        exact history reset call :func:`reset_mpm_solver_state` after authoring their state.
        """

    def check_step(self, captured: bool, debug: bool) -> None:
        """Raise asynchronous sparse-grid rebuild failures after a graph replay."""
        if (debug or self.native not in _deferred_status_checks.get()) and (
            captured or getattr(self.native, "step_status", None) is not None
        ):
            self.native.check_sparse_grid_rebuild_status()


def reset_mpm_solver_state(
    backend: NewtonBackend,
    state: State | None = None,
    world_mask: wp.array(dtype=wp.bool) | None = None,
    flags: StateFlags | int | None = None,
) -> None:
    """Reset MPM and coupled-solver history after task state is rewritten.

    When :paramref:`state` is omitted, both distinct backend state buffers are reset so a later buffer swap cannot
    restore stale history. A mask follows Newton's canonical ``world_count + 1`` contract, where the last entry
    selects global entities in world -1. A selected single local world is promoted to a full reset because a
    one-world MPM grid has no environment offsets.

    Args:
        backend: Backend whose solver to reset, e.g. :attr:`NewtonManager.backend`.
        state: State whose solver-owned history should be reset. If omitted, reset both backend states.
        world_mask: Canonical per-world mask, including the final global-world entry.
        flags: State components whose solver-owned history should reset.

    Raises:
        RuntimeError: If the MPM solver or a usable state is not initialized.
        ValueError: If :paramref:`world_mask` does not use Newton's canonical shape.
    """
    solver = backend.solver
    if not implicit_mpm_solvers(solver):
        raise RuntimeError("An implicit MPM solver is not initialized; cannot reset solver state.")
    model = backend.model

    reset_mask = world_mask
    if world_mask is not None:
        expected_shape = (model.world_count + 1,)
        if world_mask.shape != expected_shape:
            raise ValueError(f"world_mask must have shape {expected_shape}; got {world_mask.shape}.")
        if model.world_count == 1:
            selected = world_mask.numpy()
            if not selected.any():
                return
            if selected[0] and not selected[-1]:
                reset_mask = None

    candidates = (state,) if state is not None else (backend.state_1, backend.state_0)
    states = list({id(candidate): candidate for candidate in candidates if candidate is not None}.values())
    if not states:
        raise RuntimeError("Newton state is not initialized; provide an explicit state to reset.")
    check_mpm_reset_status(solver)
    for candidate in states:
        solver.reset(candidate, world_mask=reset_mask, flags=flags)
