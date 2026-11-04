# Particle NeRD physics solver

`ParticleNeRDSolverCfg` selects learned particle dynamics through the Newton
`CouplerEntryCfg` interface. The solver predicts particle positions and reconstructs
collider reaction wrenches; the existing proxy coupler transfers those wrenches to
the rigid solver. It never constructs or steps a native MPM solver, including at
initialization, reset, or particle recycling.

```python
from isaaclab_tasks.contrib.particle_nerd import ParticleNeRDSolverCfg
from isaaclab_contrib.coupling import CouplerEntryCfg

particle_entry = CouplerEntryCfg(
    name="sand",
    solver_cfg=ParticleNeRDSolverCfg(
        checkpoint="/absolute/path/to/best.pt",
        active_k="auto",
        effective_mass_scale=50.0,
        tangent_mass_scale=40.0,
    ),
    bodies=[],
    shape_label_patterns=[r"/World/ground/mpm_ground/.*"],
    all_particles=True,
    include_static_shapes=False,
    include_child_joints=False,
    substeps=1,
    in_place=True,
)
```

Use `NewtonParticleNeRDCouplerManager` as the enclosing `CouplerProxyCfg.class_type`
and map the rigid collider bodies to this entry. See the complete working
[physics configuration](../g1_29dof_particle_nerd_moving_patch/env_cfg/physics_cfg.py).
The entry factory is `NewtonParticleNeRDManager`; the dynamics implementation is
`ParticleNeRDSolver`, implementing Newton's `SolverBase` and `CouplingInterface`.
Existing particle assets and their `mpm` material-attribute namespace are reused as
storage. Those names do not enable native MPM simulation.

## Bundled checkpoint

`checkpoint/best.pt` is copied from the `g1-recurrent-05s-20261021` experiment's
`checkpoint/model/best.pt`. The fixed-terrain task resolves this file relative to
the installed task package; no machine-specific path or environment variable is
required. The checkpoint is included in package data and uses the repository's
existing Git LFS rule for `.pt` files.

SHA-256: `7eee570d75d00b4a09006e4ffd15a241510c3fd3ba74295d0396254187d00633`.

## Parameters and supported checkpoints

- `checkpoint`: required `particle_transolver_v1` checkpoint with
  `g1_particle_nerd_v1` schema and three position-history frames. The solver checks
  the timestep against the checkpoint (the current trained model uses 0.005 s).
- `collider_body_patterns`: two regular expressions identifying one box collider
  body each, per world. The current model schema is limited to two box colliders;
  arbitrary collider counts or shapes require a compatible model and features.
- `active_k`: active particle budget, or `"auto"` for material-dependent selection;
  `None` uses all valid particles. `anchor_particles` controls background features.
- `selection_interval`: physics steps between active-particle selection refreshes.
  Intermediate steps reuse particle IDs and blend weights while still running neural
  inference. The default is 1; the fixed G1 demo explicitly uses 4, matching its example.
- `batch_inference`: enabled by default. Sand worlds with matching network input
  shapes share one Transolver call. Each world retains its own material features,
  collider poses, terrain transforms, selection weights, and displacement reconstruction.
  Worlds with different input shapes form separate batches; rigid-only worlds are excluded.
  Set this solver parameter to `False` to use sequential inference for comparison or
  to reduce peak network memory. BF16 batching can introduce small floating-point differences.
- `batch_contacts`: enabled by default. Independent collider contact problems share
  one GPU launch for initialization and one for iterative reconstruction, with one
  block per collider. Each collider retains its own contact buffers, virtual mass,
  friction, and solver statistics. The contact equations, iteration limit, and
  convergence tolerance are unchanged. Set this parameter to `False` to compare
  against sequential contact solves.
- `effective_mass_scale` and `tangent_mass_scale`: dimensionless normal and
  tangential material mass multipliers for contact reconstruction. They change
  reconstructed reactions, not neural weights or the particle asset's mass.
  The virtual collider mass is a separate quantity assigned by the rigid coupler;
  body-inertia notifications refresh the contact estimator's cached scalar masses
  and invalidate solver graphs.
- `horizontal_mass_scale`, `upward_mass_scale`, and `normal_power`: directional
  factors and exponent in the normal material response.
- `contact_iterations`, `contact_relaxation`: reconstruction iterations and update
  relaxation. `gradient_radius`, `gradient_contact_band`, `contact_stabilization`,
  and `contact_offset` configure the existing contact reconstruction.
- `bed_depth`: canonical boundary depth in meters, set from the task terrain.
- `compile_model`: compile neural inference with PyTorch. Requires CUDA.

The task disables material randomization because this integration builds material
features once per world. It uses the configured particle asset material values;
changing material features at runtime is not supported. Particle storage must
have equal capacity per particle-bearing world, with contiguous world-major ordering.
Rigid-only worlds may contain no particles; the solver allocates no neural history,
inference buffers, or contact reconstruction for those worlds.

Network collider features use a snapshot of the rigid state before each coupled
step. Contact reconstruction uses the virtual proxy state after rigid integration
and lagged-wrench removal, matching the G1 example. Builder preparation preserves
the example's triangle-mesh robot collisions and kinematic boundary inertia;
this preprocessing does not instantiate a native material solver.

## Moving terrain and state

`patch/` uses the fixed particle pool and toroidal recycling of
`g1_29dof_mpm_moving_patch`. Particles that remain inside the moving patch keep
their deformation and three-frame neural position history. Particles that leave
are reused at the entering edge: terrain geometry supplies their new positions,
velocity is zeroed, and all three history frames are initialized to that position.
The outer boundary has zero mass and is pinned to terrain geometry.

There is no archive of departed particles or visited tiles. Returning to an old
region creates fresh material; old footprints outside the patch are not restored.
Storage and rendering are bounded by the current patch. Only its live neural
history is retained. Reset affects only the selected worlds.

Particle updates run as GPU kernels before the coupled solve
(`use_cuda_graph=False`). `capture_solver=True` captures coupled GPU steps using
PyTorch-owned graph memory pools. Recycling changes buffer values without changing
particle slots or shapes, so moving the patch reuses the same solver graphs. The
first requested step initializes lazy resources; recording adds no extra simulated
timestep. Network compilation can still add initial startup cost.

`network/` and `contact/` contain the extracted inference and force reconstruction
implementation; running the solver does not import the `examples/` directory.

## Validation

Set `PARTICLE_NERD_CHECKPOINT` and run:

```bash
uv run python -m pytest source/isaaclab_tasks/test/contrib/test_particle_nerd.py -q
```

The CUDA integration test blocks native MPM construction and stepping. It checks
two-world contact reactions, preserved history in the overlap, fresh history for
recycled particles, discarded old footprints, independent world resets, current
particle rendering, and CUDA graph reuse during travel.
It skips when CUDA or a checkpoint is unavailable. These checks validate solver
integration and state management; they do not measure prediction accuracy or
trained-policy robustness.
