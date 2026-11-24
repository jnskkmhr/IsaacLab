# Particle NeRD Unitree Go2 Quadruped Task

Batched manager-based environment for a 12-DoF Unitree Go2 quadruped coupled two-way with a `2.0 m x 1.2 m x 0.12 m` granular sand bed (`7,056` particles per environment).

## Registered Gym Environments

- `IsaacContrib-ParticleNeRD-Go2-v0`: Two-way coupled `ParticleNeRDSolver` + MuJoCo Warp rigid solver.
- `IsaacContrib-ParticleNeRD-Go2-MPM-v0`: Ground-truth Implicit MPM + MuJoCo Warp counterpart.

## Usage

```bash
uv run --no-sync python scripts/environments/zero_agent.py \
    --task IsaacContrib-ParticleNeRD-Go2-v0 \
    --num_envs 4 \
    --headless
```
