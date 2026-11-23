# Particle NeRD Unitree G1 Humanoid Task

Batched manager-based environment for a 29-DoF Unitree G1 humanoid with box feet coupled two-way with a `2.0 m x 1.2 m x 0.25 m` granular sand bed (`18,400` particles per environment).

## Registered Gym Environments

- `IsaacContrib-ParticleNeRDLocal-G1-v0`: Two-way coupled `ParticleNeRDSolver` + MuJoCo Warp rigid solver.
- `IsaacContrib-ParticleNeRDLocal-G1-MPM-v0`: Ground-truth Implicit MPM + MuJoCo Warp counterpart.

## Usage

Play the reference MPM policy (`5f1xyc21`):

```bash
uv run --with wandb --extra ovrtx isaaclab play \
    --rl_library rsl_rl \
    --task IsaacContrib-ParticleNeRDLocal-G1-v0 \
    --num_envs 4 \
    --wandb_run 5f1xyc21 \
    --wandb_project g1_29dof_mpm_mixed_terrain \
    --viz newton_rtx \
    "env.video_recorders=[]"
```

The policy configuration matches that checkpoint's `[512, 256, 128]` hidden layers and ordered
observations: the actor receives 930 proprioceptive history values plus seven velocity/heading
command values; the critic receives 940 proprioceptive history values, the same command, and
160 privileged history values. Histories contain ten policy samples. Policy updates run at
50 Hz and physics at 200 Hz. The local demo commands 0.25 m/s forward motion and zero yaw.
Foot-contact observations read the sand reaction wrench from the coupler.

To step without a policy:

```bash
uv run --no-sync python scripts/environments/zero_agent.py \
    --task IsaacContrib-ParticleNeRDLocal-G1-v0 \
    --num_envs 4 \
    --headless
```
