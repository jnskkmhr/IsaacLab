# Particle NeRD: methodology blocks and the code that implements them

This file maps the blocks of the methodology document (`docs/_build/particle_nerd_local_methodology.pdf`) to the
code in this package. "Eq." numbers are the equation numbers of that document. Line numbers refer to the files in
this directory unless a path is given.

How this was checked: every file, line and name below was matched against the source with a script. The text of
the methodology document was searched by section and keyword, not read end to end, and its formulas do not survive
text extraction. Rows marked *(form not checked)* say where the code for that item is; the formula itself was not
compared term by term.

## One solver step

One call of `NewtonParticleNeRDCouplerManager._step_solver` (`nerd_manager.py:173`) advances the rigid bodies
and the particles by one time step, in the staggered order of Section 1.2 of the document:

1. **Capture.** `ParticleNeRDSolver.capture_collider_state` (`nerd_solver.py:341`) copies the body poses and
   twists at the start of the step. Inside `ParticleNeRDSolver.step` (`nerd_solver.py:465`),
   `ContactWrenchSolver.capture` (`nerd_wrench_solver.py:239`) keeps the particle positions at the start of the step.
2. **Particle update (Modules 1 to 3).** `ParticleNeRDSolver.step` calls
   `ParticleNeRDNetwork.step_from_history_batched` (`network/nerd_network.py:461`), which runs the predictor and the
   input features (Module 1), the neighbourhood bases (Module 2) and the network with the gated update (Module 3).
   The solver then clamps the result to the tray and advances the position history (`nerd_solver.py:568`).
3. **Contact impulses (Module 4).** `ContactWrenchSolver.evaluate` (`nerd_wrench_solver.py:257`) uses the start-of-step
   positions and the new particle velocities to solve for the wrench on every collider.
4. **Rigid bodies (Module 5).** The coupled solver reads the wrenches through
   `ParticleNeRDSolver.coupling_harvest_proxy_wrenches` (`nerd_solver.py:360`) and applies them to the rigid
   solver over its substeps.

Training uses the single-environment twin `ParticleNeRDNetwork.step_from_history` (`network/nerd_network.py:405`), which
is differentiable. Both paths share `encode` and `_update`; they differ only in the Warp kernels that build the
inputs (one environment with gradients, or all environments at once without).

## Module 1: Kinematic predictor (Section 2)

| Methodology item | Equation(s) | File | Class / function | What it does in one line |
| --- | --- | --- | --- | --- |
| 2.1 Finite-difference velocities | Eq. (1) | `nerd_kernels.py:271` | `particle_features` | Two velocities from the last three positions. |
| 2.1 Semi-implicit predictor | Eq. (2) | `network/feature_builder.py:56` | `integrate_pos_vel` | Advances every particle under gravity alone; used only to build inputs. |
| 2.1 Position-free particle features | Eq. (3) | `nerd_kernels.py:271` | `particle_features` | 22 numbers per particle: the two velocities, the material, the step and gravity; no position. |
| 2.1 Material vector | Eq. (3) | `network/feature_builder.py:30` | `material_features` | Normalizes the 13 material and time-step entries. |
| 2.1 Same, all environments at once | Eqs. (1), (3) | `nerd_kernels.py:341` | `particle_features_batched` | Batched twin used by the solver. |
| 2.2 Collider shape pose and surface velocity | Eq. (4) | `nerd_solver.py:395` | `ParticleNeRDSolver._collider_state` | Composes body pose and shape offset; moves the body twist to the shape origin. |
| 2.2 Signed distance and outward normal | text before Eq. (4) | `nerd_kernels.py:43` | `primitive_surface` | Analytic box, sphere, cylinder, capsule and plane, with smoothed outside corners. |
| 2.2 Relative surface velocity | Eq. (4) | `nerd_kernels.py:293` | `collider_features` | Collider velocity at the closest surface point minus the particle velocity. |
| 2.2 Radius-adjusted, normalized gap | Eq. (5) | `nerd_kernels.py:293` | `collider_features` | Gap divided by the support length *(form not checked)*. |
| 2.2 Two-scale gap feature | Eq. (6) | `nerd_kernels.py:328` | `collider_features` | Near-field tanh plus a longer-range tanh, blended by `far_gap_weight`. |
| 2.2 Two-scale influence weight | Eq. (7) | `nerd_kernels.py:337` | `collider_features` | Near-field and longer-range bell curves, blended by `far_influence_weight`. |
| 2.2 Eight-channel pair feature | Eq. (8) | `nerd_kernels.py:328` | `collider_features` | Gap, normal (3), relative velocity (3), friction. |
| 2.2 Same, all environments at once | Eqs. (4)-(8) | `nerd_kernels.py:365` | `collider_features_batched` | Batched twin used by the solver. |
| Wrappers that launch the kernels | Eqs. (1)-(8) | `network/feature_builder.py:146` | `compute_features` | One environment; differentiable through `_DifferentiableFeatures` (`network/feature_builder.py:73`). |
| Wrappers that launch the kernels | Eqs. (1)-(8) | `network/feature_builder.py:233` | `compute_features_batched` | All environments, inference only. |

The long-range influence weight is divided by the number of moving colliders when there is more than one
(`ParticleNeRDNetwork._far_influence_weight`, `network/nerd_network.py:400`); whether the document states this was not
checked.

## Module 2: Local interactions (Section 3)

| Methodology item | Equation(s) | File | Class / function | What it does in one line |
| --- | --- | --- | --- | --- |
| Neighbour search and particle basis | text of Section 3 | `nerd_kernels.py:128` | `neighbor_basis_and_packing` | Hash-grid neighbours within the radius, splatted on a 4 x 4 x 4 filter grid. |
| Radial window | text of Section 3 | `nerd_kernels.py:119` | `poly6_window` | Weight that falls to zero at the neighbourhood radius *(form not checked)*. |
| Ball-to-cube mapping | text of Section 3 | `nerd_kernels.py:105` | `ball_to_cube_radial` | Maps a neighbour offset in the unit ball to the unit cube. |
| Packing signal and its rate | Eq. (9) | `nerd_kernels.py:128` | `neighbor_basis_and_packing` | Weighted neighbour count and its rate *(form not checked)*. |
| Six-channel packing state | Eq. (10) | `nerd_kernels.py:128` | `neighbor_basis_and_packing` | Count, packing, rate, mean relative velocity (3). |
| Boundary basis | text before Eq. (11) | `nerd_kernels.py:216` | `collider_basis` | Splats the pair features of Eq. (8) along the inward normal, weighted by the influence. |
| Learned embeddings | Eq. (11) | `modules/cnn.py:201` | `LocalNeighborhoodAggregator.forward` | Two linear layers turn the particle and boundary bases into 64-dimensional embeddings. |
| Launching the kernels | Eqs. (9)-(11) | `modules/cnn.py:71` | `LocalNeighborhoodAggregator.compute_bases` | One environment. |
| Launching the kernels | Eqs. (9)-(11) | `modules/cnn.py:134` | `LocalNeighborhoodAggregator.compute_bases_batched` | All environments; neighbours stay inside their environment. |

## Module 3: Neural motion update (Section 4)

| Methodology item | Equation(s) | File | Class / function | What it does in one line |
| --- | --- | --- | --- | --- |
| 4.1 Token encoding and fusion | Eq. (12) | `network/nerd_network.py:300` | `ParticleNeRDNetwork.encode` | Encodes particle features with the packing state, adds the influence-weighted collider encoding, and fuses with both embeddings (`network/nerd_network.py:333`). |
| 4.2 Slice anchors and positions | text before Eq. (13) | `modules/attention.py:88` | `PhysicsAttention.precompute_rope_cache` | Mean-centred positions, the regular slice lattice mapped to the cloud extent, rotary tables. |
| 4.2 Slice assignment | Eq. (13) | `modules/attention.py:117` | `PhysicsAttention.forward` | Feature similarity plus a spatial locality term, softmax over slices (`modules/attention.py:155`). |
| 4.2 Slice tokens and centroids | Eq. (14) | `modules/attention.py:117` | `PhysicsAttention.forward` | Weighted average of particle values and positions per slice (`modules/attention.py:162`). |
| 4.2 Slice-to-slice attention | Eqs. (15)-(16) | `modules/attention.py:117` | `PhysicsAttention.forward` | Attention between slices with RMS-normalized, rotated queries and keys (`modules/attention.py:185`). |
| 4.2 Back to particles | Eq. (17) | `modules/attention.py:117` | `PhysicsAttention.forward` | Assignment-weighted sum plus particle-to-slice attention (`modules/attention.py:194`). |
| 4.2 3D rotary embedding | Eqs. (15)-(17) | `modules/positional_encoder.py:15` | `ParticleRotaryEmbedding` | Rotation angles from continuous 3D coordinates; applied by `apply_rope` (`modules/attention.py:25`). |
| 4.2 Normalization and feed-forward | text of 4.2 | `modules/normalization.py:14` | `RMSNorm` | With `FeedForwardSwiGLU` (`modules/attention.py:31`) and `TransolverBlock` (`modules/attention.py:201`). |
| 4.3 Motion-head input | Eq. (18) | `network/nerd_network.py:351` | `ParticleNeRDNetwork.encode` | Attended token, local features, influence-weighted collider geometry, packing state. |
| 4.3 Gated velocity update | Eq. (20) | `network/nerd_network.py:397` | `ParticleNeRDNetwork._update` | `v_next = a (v + dv) + b w`, with the two sigmoid gates of `network/nerd_network.py:379`. |
| 4.3 Position update | Eq. (19) | `network/nerd_network.py:396` | `ParticleNeRDNetwork._update` | `x_next = x + dt v_next + a dx`. |
| 4.3 History advance, all particles | text after Eq. (20) | `nerd_solver.py:568` | `ParticleNeRDSolver.step` | Stores `x_next - dt v_next` as the previous position, so the stored velocity is `v_next`. |
| 4.3 History advance, active set | text after Eq. (20) | `nerd_kernels.py:518` | `scatter_active_step` | Same for the stepped particles; `advance_stationary_history` (`nerd_kernels.py:496`) handles the others. |
| 4.4 Cloud context | text of 4.4 | `network/nerd_network.py:210` | `ParticleNeRDNetwork._cloud_context` | 22 translation-invariant numbers per particle that describe the whole cloud; added to the token at `network/nerd_network.py:341`. |
| 4.4 Cylindrical skip decoding | Eqs. (21)-(22) | `network/nerd_network.py:354` | `ParticleNeRDNetwork.encode` | Second head in radial, azimuthal and vertical components, rotated to the world frame and added (`network/nerd_network.py:364`). |

## Module 4: Contact impulse solver (Section 5)

| Methodology item | Equation(s) | File | Class / function | What it does in one line |
| --- | --- | --- | --- | --- |
| Entry point | Section 5 | `nerd_wrench_solver.py:257` | `ContactWrenchSolver.evaluate` | Launches the kernels below for all colliders of all environments. |
| Body state per collider | Section 5 | `nerd_kernels.py:879` | `prepare_contact_bodies` | Surface pose, free twist and world inverse inertia. |
| 5.1 Collider surface query | text of 5.1 | `nerd_kernels.py:659` | `collider_surface` | Signed distance, normal and closest point in double precision. |
| 5.1 Velocity gradient fit | Eq. (23) | `nerd_kernels.py:905` | `prepare_contacts` | Weighted least squares over neighbours (`nerd_kernels.py:968`) *(form not checked)*. |
| 5.1 Wendland packing sum | Eq. (24) | none | none | Not in the code (see below). |
| 5.1 Velocity at the surface and candidate test | Eq. (25) | `nerd_kernels.py:905` | `prepare_contacts` | Extrapolates the velocity to the closest point (`nerd_kernels.py:970`) and keeps closing contacts (`nerd_kernels.py:974`). |
| 5.2 Packing-state mass multiplier | Eq. (26) | none | none | Not in the code (see below). |
| 5.2 Direction-dependent multipliers and effective masses | Eqs. (27)-(30) | `nerd_kernels.py:830` | `contact_mobilities` | Effective mass = particle mass times a scale chosen by the contact normal direction *(form not checked)*. |
| 5.3 Shared body velocity | Eq. (31) | `nerd_kernels.py:1042` | `initialize_body_velocity_block` | Warm start of the body twist; uses `contact_response` (`nerd_kernels.py:756`) and `linear_solve_6` (`nerd_kernels.py:796`). |
| 5.3 Impulse iterations | Eq. (32) | `nerd_kernels.py:1182` | `solve_contact_block` | Relaxed Jacobi iterations, normal impulse then friction (`friction_disk`, `nerd_kernels.py:725`). |
| 5.3 Wrench accumulation | Eq. (33) | `nerd_kernels.py:1286` | `solve_contact_block` | Sum of impulses and their moments, divided by the time step. |
| Bookkeeping | Section 5 | `nerd_kernels.py:1019` | `compact_contacts` | Packs the flagged contacts; `gather_active_particles` (`nerd_kernels.py:858`) limits the search to the stepped particles. |

## Module 5: Rigid-body integration (Section 6)

| Methodology item | Equation(s) | File | Class / function | What it does in one line |
| --- | --- | --- | --- | --- |
| Hand the wrench to the rigid solver | Section 6 | `nerd_solver.py:360` | `ParticleNeRDSolver.coupling_harvest_proxy_wrenches` | Writes the wrenches to the coupled body-force array through `scatter_wrench` (`nerd_kernels.py:611`). |
| Remove the previous wrench from the proxy bodies | Section 6 | `nerd_solver.py:351` | `ParticleNeRDSolver.coupling_rewind_proxy_body` | Undoes the lagged wrench on the body velocities the particle solver sees (`rewind_proxy`, `nerd_kernels.py:590`). |
| Step both solvers | Section 6 | `nerd_manager.py:173` | `NewtonParticleNeRDCouplerManager._step_solver` | Steps the coupled solver, optionally as a recorded CUDA graph. |
| Build the particle solver | Section 6 | `nerd_manager.py:102` | `NewtonParticleNeRDManager._create_solver` | Creates `ParticleNeRDSolver` for the `sand` entry of the coupler. |

The rigid solver itself (MuJoCo Warp, two substeps, lagged proxy coupling) is chosen in the task files
`example/*/env_cfg/physics_cfg.py`; the coupling logic is `isaaclab_contrib.coupling`, not this package.

## Section 7: Training objective

Training is not in this package. It lives in `scripts/neural/` (`train.py`, `losses.py`, `rollout.py`). None of
the rows below was compared with the formulas of the document.

| Methodology item | Equation(s) | Where in `scripts/neural/` | What it does in one line |
| --- | --- | --- | --- |
| Recurrent rollout without teacher forcing | Section 7 | `rollout` (`rollout.py`, line 22) | Feeds the model its own predictions for the whole window. |
| Position, velocity and rest losses | Eqs. (34)-(36) | `world_particle_loss` (`losses.py`, line 32), `balanced_loss` (`losses.py`, line 16) | `0.6 position + 0.25 velocity + 0.15 rest drift`. |
| Unconfined-cloud terms | Eqs. (37)-(41) | none | Not in the code (see below). |
| Window curriculum | Section 7 | `main` (`train.py`, lines 272-273) | Six equal stages of 8, 16, 32, 64, 100, 100 steps, capped at `--horizon`. |
| Long resting rollouts | not found in the document | `rest_unroll` (`losses.py`, line 55) | Rolls a resting bed out for `--rest-steps` steps in segments (`--rest-fraction` of the training steps). |

## In the code but not in the methodology document

Searched for by keyword in the text of the document; none was found.

| Item | Where | What it does |
| --- | --- | --- |
| Particle sleeping (`sleep_speed`) | `nerd_solver.py:560`, `nerd_kernels.py:549` | A particle whose predicted move in a step is shorter than `sleep_speed * dt` keeps its position and gets zero velocity. Off by default. |
| Active-particle selection | `ParticleNeRDSolver._select_active` (`nerd_solver.py:425`), `active_particle_score` (`nerd_kernels.py:438`) | Steps only the K particles with the lowest score (gap to the nearest collider minus a speed term) and keeps the others in place; reselected every `selection_interval` steps. Off by default. |
| Weighted background anchors | `nerd_solver.py:454`, `multiplicity` in `PhysicsAttention.forward` (`modules/attention.py:117`) | Up to 256 evenly spaced other particles are added as input only; each carries a weight so that together they stand for all particles that are not stepped. |
| Blend at the edge of the active set | `nerd_solver.py:456`, `scatter_active_step` (`nerd_kernels.py:518`) | Particles that barely made the active set get a partial step, so that the set changes smoothly. |
| Tray clamp | `nerd_solver.py:559`, `scatter_active_step` (`nerd_kernels.py:518`) | Predicted positions are clamped to the inside of the tray. |
| Compiled network and recorded step | `nerd_solver.py:296`, `nerd_manager.py:173` | `torch.compile` of `encode` and CUDA-graph replay of the coupled step (speed only). |
| Checkpoint loader | `ParticleNeRDNetwork.load_checkpoint` (`network/nerd_network.py:185`) | Ignores the weights of removed modules, rejects removed options. |

The bed settle and relax step of the earlier package is no longer in the code. No task enabled it.

## In the methodology document but not in the code

| Item | Document | State of the code |
| --- | --- | --- |
| Wendland packing sum in the wrench solver | Eq. (24) | Not computed. The earlier kernel accumulated it and never used it; that accumulation was removed with the dead code. |
| Packing-state effective-mass multiplier | Eq. (26) | Not applied. `contact_mobilities` (`nerd_kernels.py:830`) takes the contact normal, the particle mass and the configured scales only, so the multiplier is effectively one and only the direction-dependent scaling of Eqs. (27)-(30) acts. The "packing- and direction-dependent mobilities" of Section 1.2 are therefore direction-dependent only. |
| Signed-distance-field colliders | Section 2.2 ("trilinearly interpolated 3D signed-distance fields") | Removed. Every task had the option switched off and training never used it. Colliders are the analytic shapes of `primitive_surface`. |
| Loss terms for unconfined particle clouds | Section 7.1, Eqs. (37)-(41) | Not in `scripts/neural`. The earlier trainer had them, but only for a record shape that the current dataset format never produces, so no run on the current dataset used them. Free-fall episodes are trained with the loss of Eqs. (34)-(36). |

Removed in this clean-up and not described in the document (keyword search): the ballistic update rule, the stress
contact head, and the model's own top-K selection.

## Solver settings and the block each one affects

Fields of `ParticleNeRDSolverCfg` (`nerd_solver_cfg.py`).

| Field | Default | Block |
| --- | --- | --- |
| `checkpoint` | `checkpoint/best.pt` | Modules 1 to 3: weights, time step, support length and the far-field weights of Eqs. (6)-(7). |
| `collider_body_patterns` | `[".*/Object"]` | Modules 1 and 4: which bodies are colliders. |
| `bed_depth` | 0.3 m | Module 1: placement of the tray boxes when the scene has no static boxes. |
| `active_k` | `None` | Not in the document: number of stepped particles per environment. |
| `selection_interval` | 4 | Not in the document: steps between two selections of the active set. |
| `sleep_speed` | 0.0 m/s | Not in the document: particle sleeping (0.01 recommended by the benchmark). |
| `compile_model` | `True` | Module 3, speed only. |
| `capture_solver` | `True` | Module 5, speed only: record and replay the coupled step. |
| `effective_mass_scale` | 10.0 | Module 4, Section 5.2: mass scale when the collider pushes the particle down. |
| `horizontal_mass_scale` | 1.0 | Module 4, Section 5.2: mass scale for a horizontal contact normal. |
| `upward_mass_scale` | 1.0 | Module 4, Section 5.2: mass scale when the collider lifts the particle. |
| `tangent_mass_scale` | 10.0 | Module 4, Section 5.2: mass scale for friction. |
| `normal_power` | 1.0 | Module 4, Section 5.2: exponent of the blend between the three normal directions. |
| `contact_iterations` | 200 | Module 4, Section 5.3: maximum number of Jacobi iterations. |
| `contact_relaxation` | 0.02 | Module 4, Section 5.3: relaxation factor. |
| `gradient_contact_band` | 0.08 m | Module 4, Section 5.1: distance within which a particle is a contact candidate. |
| `contact_stabilization` | 0.01 | Module 4, Section 5.3: share of the penetration removed per step. |
| `contact_offset` | 0.0 m | Module 4, Section 5.1: extra margin on the collider surface. |

The neighbour radius of Module 2 and of the velocity-gradient fit (4.5 particle radii) and the support length of
Eq. (5) (4.375 particle radii, at least 0.035 m) are set from the particle radius of the scene in
`ParticleNeRDSolver.__init__` (`nerd_solver.py:129`, the two assignments at `nerd_solver.py:271`), not from the
configuration.
