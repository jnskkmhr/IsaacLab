# Transolver, active compaction, and batched simulation

This document describes the current `ParticleTransolver` (`particle_transolver_v1`) architecture, quadrature-weighted active particle compaction, and analytical/RFT contact-force estimation across Box, Shovel, and G1.

## Source map

| Task | Source |
| --- | --- |
| Particle model and box/shovel training | `transolver/model.py`, `transolver/features.py`, `transolver/train.py`, `transolver/evaluate.py` |
| Single-intruder coupled dynamics | `runner/simulation.py`, `runner/play.py`, `transolver/closed_loop.py` |
| Batched multi-environment box/shovel simulation | `examples/particle_nerd_batch/` (`model.py`, `features.py`, `state.py`, `simulation.py`, `force/`) |
| G1 collection, training, active compaction, and dynamic evaluation | `g1/collect.py`, `g1/train.py`, `g1/data.py`, `g1/adapter.py`, `g1/forces.py`, `g1/play.py`, `g1/evaluate.py` |

## Particle Transolver inputs and outputs

For `N` particles, `C` colliders, and `H = 3` position-history frames:

| Input | Shape | Definition |
| --- | --- | --- |
| Particle features | `[N, 25]` | Current particle position relative to configured origin (3); two finite-difference velocities (6); ten normalized material parameters plus particle mass/radius (12); timestep (1); gravity (3) |
| Collider pair features | `[N, C, 8]` | Signed particle-to-collider surface gap / support (1), outward world normal (3), collider surface velocity minus particle velocity (3), contact friction (1) |
| Collider influence | `[N, C]` | `max(0, 1 - max(gap, 0) / support)` with default support `0.07 m` |
| Quadrature multiplicity | `[N]` (optional) | `1.0` for the `K` active particles and `(N - K) / M_valid` for the `M` strided background anchors |
| Particle output | `[N, 3]` | Displacement in metres (`x_next = x + displacement`, `v_next = displacement / dt`) |

## Quadrature-Weighted Active Particle Compaction

- Naive top-`K` selection alters the slice-attention normalization denominator `sum_j w_{j,s} + 0.01` inside `PhysicsAttention`.
- Passing `multiplicity` weights the `M` strided background anchors by `(N - K) / M_valid` so slice token averages match the full `N`-particle bed while only evaluating `K + M << N` particles.
- Both the `K` active particles (with outer-rim `C^1` Hermite taper and free-surface ballistics) and the `M` background anchors update their history recurrently, keeping background velocity histories in-distribution over multi-second rollouts.
