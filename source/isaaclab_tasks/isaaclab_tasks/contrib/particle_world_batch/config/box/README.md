# Box Two-Way Coupling (`contrib/particle_world_batch/config/box`)

Standalone manager-based RL environment modeled after Newton's
[`example_mpm_twoway_coupling.py`](https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_twoway_coupling.py).

- **Task IDs**:
  - `IsaacContrib-ParticleWorld-Box-v0` (Neural `WorldParticle` solver)
  - `IsaacContrib-ParticleWorld-Box-MPM-v0` (Reference `SolverImplicitMPM`)
- **Physical Setup**:
  - Granular sand bed (`density = 2500 kg/m³`, `friction = 0.75`) inside a rigid retaining tray.
  - Dynamic rigid cuboid (`mass = 75 kg`) dropped onto the sand bed with two-way lagged proxy coupling (`substeps = 4` rigid per sand step).

## Run

```bash
uv run isaaclab -p scripts/environments/zero_agent.py \
  --task IsaacContrib-ParticleWorld-Box-v0 \
  --num_envs 1 \
  --viz newton_gl
```
