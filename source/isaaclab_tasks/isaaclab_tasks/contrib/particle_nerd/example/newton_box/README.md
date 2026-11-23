# Particle NeRD Newton Box Task

Passive scene replicating Newton's `example_mpm_twoway_coupling`: six 75 kg boxes (friction 0.5) dropped from `z = 2.0 m`, `0.6 m` apart vertically, onto a `2 m x 2 m x 0.5 m` sand bed on a ground plane at `z = 0`. Box sizes are the example's half extents doubled.

Deviations from the example: the sand material is the Particle NeRD training material (friction 0.48, density 1600 kg/m^3, the network takes material as input) instead of friction 0.75 and 2500 kg/m^3; the particle spacing is `0.032 m` (the Particle NeRD Local training resolution, about 63k particles) instead of 3 particles per `0.05 m` voxel (about 430k), the sim step is `5 ms` instead of four `2.5 ms` substeps per `10 ms` frame, and the ground is a large slab with hidden boundary walls 10 m away so the solver's floor-and-four-walls layout is satisfied. The sand bed is not confined, so it spreads at its edges as in the example.

The Particle NeRD Local network was trained with a single rigid collider. Six boxes run mechanically, but accuracy is not validated.

## Registered Gym Environments

- `IsaacContrib-ParticleNeRDLocal-NewtonBox-v0`: Particle NeRD Local + MuJoCo Warp.
- `IsaacContrib-ParticleNeRDLocal-NewtonBox-MPM-v0`: implicit MPM + MuJoCo Warp.

## Usage

```bash
uv run --no-sync python scripts/environments/zero_agent.py \
    --task IsaacContrib-ParticleNeRDLocal-NewtonBox-v0 \
    --num_envs 1 \
    --visualizer newton_gl
```
