# Changing a particle model

Read [transolver_and_batch.md](transolver_and_batch.md) for the explicit feature contract and source map. The checkpoint architecture is `particle_transolver_v1`:

```text
ParticleTransolver.forward(particles, colliders, influence, multiplicity=None) -> displacement [..., N, 3]
next_position = current_position + displacement
next_velocity = displacement / checkpoint_dt
```

The optimized sibling `particle_nerd_batch/model.py` supports `[E, N, ...]` batched inputs. Attention remains independent across environments. The single-environment/training implementation (`examples/particle_nerd/transolver/model.py`) and the batched implementation (`examples/particle_nerd_batch/model.py`) use compatible parameter names and state dicts; update both together when changing model semantics.

| Change | Owners to inspect together |
| --- | --- |
| Particle features or collider geometry | `transolver/features.py`, `transolver/inference.py`, `g1/data.py`, sibling `particle_nerd_batch/features.py`, `io.py`, `sdf.py` |
| Neural architecture/decoder or active compaction | `transolver/model.py`, `g1/data.py`, sibling `particle_nerd_batch/model.py`, `features.py`, checkpoint loaders and config |
| Analytical / RFT contact force estimation | `force_estimation/estimator.py`, `force_estimation/kernels.py`, `force_estimation/body_velocity.py`, `g1/forces.py`, sibling `particle_nerd_batch/force/` |
| Recurrent training | `transolver/train.py` for box/shovel; `g1/train.py` for G1 |
| Full batch step / CUDA graph capture | sibling `particle_nerd_batch/simulation.py`, `state.py`, `features.py`, `force/`, `rigid.py` |

## Validation sequence

```bash
export PYTHONPATH=/home/jkamohara/isaac/IsaacLab-particle-nerd/examples
/home/jkamohara/isaac/IsaacLab-particle-nerd/.venv/bin/pytest examples/particle_nerd/tests examples/particle_nerd_batch/tests -q
```
