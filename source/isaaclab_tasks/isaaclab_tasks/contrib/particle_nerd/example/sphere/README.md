# Particle NeRD Sphere Task

Batched manager-based environment for a dynamic rigid sphere (`radius = 0.10 m`, `2.0 kg`) coupled two-way with a `2.0 m x 1.2 m x 0.25 m` granular sand bed (`18,400` particles per environment).

## Registered Gym Environments

- `IsaacContrib-ParticleNeRDLocal-Sphere-v0`: Two-way coupled `ParticleNeRDSolver` + MuJoCo Warp rigid solver.
- `IsaacContrib-ParticleNeRDLocal-Sphere-MPM-v0`: Ground-truth Implicit MPM + MuJoCo Warp counterpart.

## Usage

```bash
uv run --no-sync python scripts/environments/zero_agent.py \
    --task IsaacContrib-ParticleNeRDLocal-Sphere-v0 \
    --num_envs 4 \
    --headless
```
