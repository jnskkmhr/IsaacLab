# Particle NeRD

This is the maintained source directory for `ParticleTransolver`, active particle compaction (`K` active particles + `M` quadrature anchors), the GPU analytical/RFT contact-force estimator, and the IsaacLab data collectors and dynamic runners for Box, Shovel, and G1.

For teammates and coding agents, start with the [agent handoff](agent/README.md) for implementation context, architecture details, and validation boundaries.

## Source layout

| Directory | Responsibility | Main entry points |
| --- | --- | --- |
| `data_collection/` | Dynamic MPM–MJWarp recordings, scene configuration, manager terms, episode reader | `collect.py`, `campaign.py`, `dataset.py` |
| `transolver/` | `ParticleTransolver`, Warp particle/collider features, recurrent training, particle inference, open-loop and closed-loop evaluation | `model.py`, `features.py`, `train.py`, `inference.py`, `evaluate.py`, `closed_loop.py` |
| `force_estimation/` | GPU velocity-gradient fitting and analytical/RFT contact impulse solve | `estimator.py`, `kernels.py`, `body_velocity.py` |
| `runner/` | Connect `ParticleTransolver` to `ContactEstimator` and dynamic MJWarp rigid-body integration | `simulation.py`, `rigid.py`, `play.py`, `viewer.py` |
| `shovel/` | Concave trowel/shovel SDF and USD asset generation | `sdf.py`, `asset.py`, `source_geometry.py` |
| `g1/` | G1 humanoid sand-bed data collection, recurrent training, active compaction, analytical virtual-foot proxy forces, and closed-loop evaluation | `collect.py`, `train.py`, `data.py`, `adapter.py`, `forces.py`, `play.py`, `evaluate.py`, `render.py` |
| `osmo/` | Package this worktree and run collection/training in OSMO | `stage.py`, `collection.yaml`, `transolver_training.yaml`, `shovel_collection.yaml`, `shovel_training.yaml` |
| `tests/` | Contact mechanics, Transolver symmetry/gradients, rigid motion, and GPU integration checks | `test_transolver.py`, `test_gpu_interfaces.py`, `test_shovel.py`, `test_g1_contact_loss.py` |

## Interfaces and step order

- `Predictor.step(body_pose, body_velocity)` (`transolver/inference.py`) takes current collider poses and velocities as CUDA tensors and updates particle position, velocity, and position history on the GPU.
- `ContactEstimator.estimate(position, next_velocity, body_pose, body_velocity, external_wrench)` (`force_estimation/estimator.py`) takes Warp GPU arrays and returns a `[1]` material-on-body wrench: world force in N followed by world torque in N m about the body center of mass.
- `CoupledSimulation.step(external_wrench=None)` (`runner/simulation.py`) predicts particles, estimates the material wrench, and advances the dynamic rigid body over four 5 ms MJWarp substeps per 20 ms particle step.
- For batched multi-environment Box/Shovel simulation with full-step CUDA graph capture, see the sibling [`examples/particle_nerd_batch`](../particle_nerd_batch/README.md) package.
- For 200 Hz coupled G1 humanoid simulation on granular sand, see [`examples/particle_nerd/g1/README.md`](g1/README.md).

## Commands

Run from the IsaacLab worktree root:

```bash
RUN=examples/particle_nerd/run.sh

# Train ParticleTransolver on Box or Shovel trajectories (0.5 s recurrent horizon)
"$RUN" -m particle_nerd.transolver.train \
  --data /path/to/dataset --output /path/to/transolver-model \
  --steps 10000 --rollout-seconds 0.5 --loss-mode balanced

# Open-loop particle rollout evaluation on recorded rigid-body motion
"$RUN" -m particle_nerd.transolver.evaluate \
  --data /path/to/dataset --checkpoint /path/to/transolver-model/best.pt \
  --output /path/to/transolver-evaluation --mode all --steps 250

# Interactive or headless coupled dynamic simulation (Box or Shovel)
"$RUN" -m particle_nerd.runner.play \
  --checkpoint /path/to/transolver-model/best.pt \
  --episode /path/to/episode.hdf5 --contact-solver rft_body_velocity

# Unit tests
"$RUN" -m pytest examples/particle_nerd/tests examples/particle_nerd_batch/tests -q
```
