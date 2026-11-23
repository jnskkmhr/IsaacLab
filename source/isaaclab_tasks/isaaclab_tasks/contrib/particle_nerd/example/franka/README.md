# Particle NeRD Franka Sand-Pushing Task

Batched manager-based environment for a 7-DoF Franka Emika Panda manipulator coupled two-way with a `1.2 m x 1.0 m x 0.25 m` granular sand bed (`9,360` particles per environment).

## Registered Gym Environments

- `IsaacContrib-ParticleNeRDLocal-Franka-v0`: Two-way coupled `ParticleNeRDSolver` + MuJoCo Warp rigid solver.
- `IsaacContrib-ParticleNeRDLocal-Franka-MPM-v0`: Ground-truth Implicit MPM + MuJoCo Warp counterpart.

## Usage

```bash
uv run --no-sync python scripts/environments/zero_agent.py \
    --task IsaacContrib-ParticleNeRDLocal-Franka-v0 \
    --num_envs 4 \
    --headless
```
