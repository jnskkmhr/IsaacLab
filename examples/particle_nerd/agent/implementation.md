# Current implementation

Read alongside [the example README](../README.md). All quantities below describe the consolidated box pipeline unless explicitly marked G1.

## Data collection

Owners: [collector](../data_collection/collect.py), [campaign](../data_collection/campaign.py), [scene configuration](../data_collection/scene_cfg.py), [manager terms](../data_collection/terms/), [episode reader](../gnn/data.py).

| Quantity | Current collector/campaign |
| --- | --- |
| Material volume at initialization | 1.0 x 1.0 x 0.5 m; rigid ground at world z=0; no retaining walls |
| MPM grid spacing | 0.04 m |
| Material points | 16,384; 32 x 32 x 16 initial lattice; spacing 0.03125 m; jitter fraction 0.2 |
| Dynamic box size | 0.12 x 0.12 x 0.08 m |
| Box mass | Single-episode default 7.5 kg; campaign samples 5–10 kg |
| Material | Sand: density 1600 kg/m³, Young's modulus 1e15 Pa, Poisson ratio 0.3, friction 0.48, yield pressure 1e15 Pa; other configured yield/hardening/dilatancy/viscosity terms zero |
| Rigid contact friction | 0.6; distinct from sand constitutive friction |
| Reference physics interval | 0.005 s; maximum 25 MPM solver iterations |
| Particle recording interval | 0.02 s, or 50 Hz |
| Episode | Settling followed by default 5 s disturbance window |
| Campaign disturbances | Rest, sustained push, continuous dragging; randomized motion and mass |

The box is dynamic. Dragging applies an external wrench; it does not prescribe the box's trajectory. The current campaign runs one environment per process. Distributed collection uses multiple campaign processes.

HDF5 contains particle position/velocity, mass/radius, rigid pose/velocity, measured contact wrenches, and a JSON `manifest`. `frame_steps` maps each particle frame to the higher-rate rigid recording. Do not index rigid arrays directly with the particle frame number. Campaign sidecars `*.campaign.json` provide episode IDs and train/validation/test assignments. Force labels are not GNN training targets.

Training samples one reference transition at a time. History comes from reference particle velocities, not a recurrent predicted rollout. Sampling favors post-settle frames but also includes settling. Gaussian velocity-history noise is applied during training. The current dataset has sand and a fixed box; material features alone do not establish generalization to snow, clay, or new geometry.

## Particle graph and GNN inputs

Owners: [GPU features](../gnn/features.py), [network](../gnn/model.py).

Let N be the material-particle count, E the allocated edge count, H the velocity-history length, and R the neighborhood radius (default 0.07 m). Positions and vectors use world coordinates. Box pose is position followed by quaternion x/y/z/w. Box velocity is linear velocity followed by angular velocity.

| Network argument | Shape | Contents |
| --- | --- | --- |
| `nodes` | N x (3H+17); default N x 26 | H velocity vectors; 13 material/sampling scalars; gravity direction; neighbor count / 64 |
| `edges` | E x 7 | Relative particle position / R (3), distance / R (1), relative velocity (3) |
| `boundary` | N x 2 x 9 | Per-particle ground and box geometry/motion features |
| `influence` | N x 2 x 1 | Distance weight for each particle–collider pair |
| `src`, `dst` | E each | Directed particle-edge indices; source sends to destination |
| `degree` | N | Neighbor count clamped to at least one, used for aggregation normalization |
| `valid` | E | Mask for occupied edge slots |

The 13 scalars are density, Young's modulus, Poisson ratio, material friction, yield pressure, tensile yield ratio, yield stress, hardening, dilatancy, viscosity, particle mass, particle radius, and timestep. Their exact normalization is in `material_features()`. Absolute particle positions are used for geometry and graph construction, not directly included in `nodes`. Stress and deformation history are not explicit inputs.

Each nine-value collider feature contains signed distance / R clipped to [-2,2], outward normal (3), collider surface velocity minus particle velocity (3), inverse body mass, and rigid contact friction (currently hardcoded 0.6). Ground inverse mass is zero. The box surface velocity includes translation and rotation at the closest point. Geometry is an analytical plane and a fixed-size box SDF; there are no sampled rigid-surface nodes.

```text
Particle features (26) --> Particle encoder -----------------------+
Box features (9) -------> Collider encoder --> x distance weight ---+--> SUM
Ground features (9) ----> Collider encoder --> x distance weight ---+     |
                                                                        v
                                                         Particle node embedding (64)
                                                                        |
Particle edges (7) -----> Edge encoder --------------------------> Message passing
```

**Collider information goes into each material-particle node embedding.** Box and ground use the collider encoder with shared parameters. There are no collider graph nodes or particle-to-collider graph edges. The distance weight is `max(0, 1 - max(distance, 0) / R)`: one on/inside the collider, zero at or beyond R outside it.

A Warp hash grid rebuilds particle neighborhoods each step. Self edges are excluded. Inference allocates N x 96 edge slots by default; this is a total capacity, not permission to truncate neighbors. Overflow invalidates the step. Padded messages are masked, but their neural computation still costs time. Training may compact edges with a host count read; inference does not.

## Current network output and training

The network uses 64 hidden channels and three message-passing blocks by default. Encoders use Linear–SiLU–Linear–LayerNorm. Each block updates edge embeddings, sums messages at destination particles, divides by square root of degree, and updates node embeddings with a residual connection.

The decoder produces six values per particle, scaled by 0.2. Both three-component corrections have units m/s:

```text
a, b = network output split into two vectors
next_position = position + dt * (current_velocity + a)
next_velocity = current_velocity + b
dt = 0.02 s
```

`a` is a displacement-rate correction, not acceleration. There is no separate gravity integration or geometric projection in this particle update. The learned corrections include material and contact response. The requested displacement-only replacement is described in [model changes](model_changes.md).

[Training](../gnn/train.py) minimizes a weighted mean of `squared_position_error / dt² + 0.5 * squared_velocity_error`. Weight is `1 + 3 * box_influence + indicator(reference_speed > 0.05 m/s)`. Defaults: AdamW, learning rate 3e-4, weight decay 1e-6, gradient clipping 1, velocity noise 0.0005 m/s, bfloat16 autocast. Checkpoint selection averages global, near-box, and moving-particle one-step position RMSE over available metrics. This is not rollout-based training or checkpoint selection.

[Optional activity selection](../gnn/region_selection.py) freezes selected particle updates. It still executes the full motion graph and is not currently a GNN-compute reduction. Omit `--selector` for the base motion model. Oracle selector modes use future reference information and are evaluation tools only.

## Analytical contact-force estimation

Owners: [estimator interface](../force_estimation/estimator.py), [Warp kernels](../force_estimation/kernels.py).

Inputs are current particle positions, predicted next particle velocities, current rigid pose/velocity, particle masses, rigid geometry/inertia, and known external wrench. Future rigid acceleration is not an input. Output is **material-on-box world force followed by world torque about the box center of mass**. Force is N; torque is N m. Gravity and known external loads are not included in the returned material wrench.

1. Build a particle hash grid and fit local velocity gradients from predicted next particle velocities.
2. Compute box free motion under gravity, known external forces/torques, and gyroscopic terms, excluding material contact.
3. Construct contact geometry and estimate material velocity at the contact locations using the local fit.
4. Iteratively solve contact impulses with unilateral normal contact, Coulomb friction, and penetration stabilization. Contacts share the box velocity response.
5. Sum contact impulses divided by dt into force; sum their moments divided by dt into torque.

For an impulse j acting on the box, the auxiliary material response subtracts `j / (effective_mass_scale * particle_mass)` from the fitted material velocity. This effective-mass approximation supplies a relationship between impulse and material velocity change. It is not a reconstruction of all particle–particle internal forces or a complete constitutive law. Accurate particle velocities therefore do not guarantee unique or exact body wrenches.

Current box defaults: effective mass scale 10, fit radius 0.065 m, stabilization 0.01, maximum 2000 contact solver iterations with early termination at tolerance 1e-9. The runner uses friction from the recording (fallback 0.6); the standalone estimator constructor defaults to 0.48. These settings have not been accuracy-calibrated for the current coupled 50 Hz model.

The estimator can represent box/sphere/cylinder geometry, but the current GNN and coupled runner support only the fixed box. Separate per-body solves would not automatically handle shared particle response or articulated coupling correctly.

## Coupled step and GPU ownership

Owner: [CoupledSimulation](../runner/simulation.py).

```text
At t: particle position/history + current dynamic box pose/velocity
  -> GNN predicts particle position and velocity at t + 20 ms
  -> estimator uses OLD particle position + PREDICTED next velocity
     + current box state + known external wrench
  -> material-on-box wrench
  -> MJWarp advances the box with that wrench and external wrench
     over four 5 ms substeps
  -> next step uses this new box state and the predicted particles
```

Only the settled initial state/history comes from the episode file in `runner.play`. Predicted particle velocities already include interaction: auxiliary contact-solver particle responses are not applied to them a second time. The material wrench is held constant during the four rigid substeps. This is explicit coupling, without a GNN/rigid-body correction loop.

MJWarp handles direct box–rigid-ground contact. That direct contact is absent from the estimator's free-motion calculation, so simultaneous box–ground and box–sand contact is a limitation. Warp/Torch streams and device buffers must remain synchronized. Only the MJWarp substeps are CUDA graph captured; `torch.compile` does not mean the entire simulation is captured. Diagnostics, viewing, loading, and recording may synchronize with the CPU.

## Separate G1 force benchmark

The G1 benchmark still updates material with MPM in closed loop using the robot's actual estimated-force-driven motion. It does not run this particle GNN. Robot actuation, gravity, and joint constraints belong in the articulated free-motion response. The existing proxy experiment uses independent virtual foot responses rather than a full generalized contact response `J * M_inverse * J_transpose`.

Its feedback timing and parameters differ from the box runner: estimated material wrench replaces the coupled feedback for a subsequent coupling step. Do not copy the box's free-body integration into each articulated link or integrate actuator forces twice. Read `proxy_estimator.py` and the coupler integration in the separate G1 worktree before changing this benchmark. Filtering experiments and their robot-motion results do not validate the current box GNN.
