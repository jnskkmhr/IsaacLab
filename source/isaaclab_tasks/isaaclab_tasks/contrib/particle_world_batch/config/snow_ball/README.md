# Snow Ball Rolling (`contrib/particle_world_batch/config/snow_ball`)

Standalone manager-based RL environment modeled after Newton's
[`example_mpm_snow_ball.py`](https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_snow_ball.py).

- **Task IDs**:
  - `IsaacContrib-ParticleWorld-SnowBall-v0` (Neural `WorldParticle` solver)
  - `IsaacContrib-ParticleWorld-SnowBall-MPM-v0` (Reference `SolverImplicitMPM`)
- **Physical Setup**:
  - Sloped cohesive snow bed with exact snow rheology (`density = 400 kg/m³`, `young_modulus = 1.4e6 Pa`, `poisson_ratio = 0.3`, `friction = 0.5`, `damping = 0.01`, `yield_pressure = 1.4e6 Pa`, `tensile_yield_ratio = 0.2`, `hardening = 5.0`, `dilatancy = 1.0`).
  - Rolling spherical snow-ball collider initialized at the top of the slope with two-way lagged proxy coupling.

## Run

```bash
uv run isaaclab -p scripts/environments/zero_agent.py \
  --task IsaacContrib-ParticleWorld-SnowBall-v0 \
  --num_envs 1 \
  --viz newton_gl
```
