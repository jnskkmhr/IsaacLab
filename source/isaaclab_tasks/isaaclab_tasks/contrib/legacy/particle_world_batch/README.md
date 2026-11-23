# Batched WorldParticle & MPM Manager-Based Environments (`contrib/legacy/particle_world_batch`)

Standalone manager-based Isaac Lab RL environments modeled directly after the canonical
[Newton MPM standalone examples](https://github.com/newton-physics/newton/tree/main/newton/examples/mpm)
so researchers can benchmark simulation quality, two-way rigid–particle coupling, and cohesive
snow/granular rheology side-by-side in both `WorldParticle` (`ParticleWorldSolverCfg`) and
native `Implicit MPM` (`MPMSolverCfg`).

---

## Task Configurations

1. **`config/box` (`IsaacContrib-ParticleWorld-Box-v0`, `IsaacContrib-ParticleWorld-Box-MPM-v0`)**:
   - Modeled after [`newton/examples/mpm/example_mpm_twoway_coupling.py`](https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_twoway_coupling.py).
   - Spawns a `2.0 m × 2.0 m × 0.50 m` granular bed (`voxel_size = 0.05 m`, `density = 2500 kg/m³`, `friction = 0.75`) with rigid box colliders (`mass = 75 kg`) interacting via two-way lagged proxy coupling.
2. **`config/snow_ball` (`IsaacContrib-ParticleWorld-SnowBall-v0`, `IsaacContrib-ParticleWorld-SnowBall-MPM-v0`)**:
   - Modeled after [`newton/examples/mpm/example_mpm_snow_ball.py`](https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_snow_ball.py).
   - Spawns a tilted $45^\circ$ slope bed with cohesive snow rheology (`density = 400 kg/m³`, `young_modulus = 1.4e6 Pa`, `poisson_ratio = 0.3`, `friction = 0.5`, `damping = 0.01`, `yield_pressure = 1.4e6 Pa`, `tensile_yield_ratio = 0.2`, `hardening = 5.0`, `dilatancy = 1.0`) and a rolling snow-ball collider at the top of the slope.

---

## Running via `uv run isaaclab`

```bash
# Box Two-Way Coupling (WorldParticle neural solver)
uv run isaaclab -p scripts/environments/zero_agent.py \
  --task IsaacContrib-ParticleWorld-Box-v0 \
  --num_envs 1 \
  --viz newton_gl

# Box Two-Way Coupling (Native Implicit MPM reference)
uv run isaaclab -p scripts/environments/zero_agent.py \
  --task IsaacContrib-ParticleWorld-Box-MPM-v0 \
  --num_envs 1 \
  --viz newton_gl

# Snow Ball Rolling (WorldParticle neural solver)
uv run isaaclab -p scripts/environments/zero_agent.py \
  --task IsaacContrib-ParticleWorld-SnowBall-v0 \
  --num_envs 1 \
  --viz newton_gl

# Snow Ball Rolling (Native Implicit MPM reference)
uv run isaaclab -p scripts/environments/zero_agent.py \
  --task IsaacContrib-ParticleWorld-SnowBall-MPM-v0 \
  --num_envs 1 \
  --viz newton_gl
```

---

## Directory Layout (following `skills/env_construction.md`)

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/legacy/particle_world_batch/
├── __init__.py
├── README.md
└── config/
    ├── __init__.py
    ├── box/
    │   ├── __init__.py
    │   ├── box_env_cfg.py
    │   ├── README.md
    │   ├── agents/
    │   │   ├── __init__.py
    │   │   ├── rsl_rl_ppo_cfg.py
    │   │   └── compositional_rsl_rl_cfg.py
    │   ├── env_cfg/
    │   │   ├── __init__.py
    │   │   ├── action_cfg.py
    │   │   ├── commands_cfg.py
    │   │   ├── curriculum_cfg.py
    │   │   ├── event_cfg.py
    │   │   ├── observation_cfg.py
    │   │   ├── physics_cfg.py
    │   │   ├── reward_cfg.py
    │   │   ├── scene_cfg.py
    │   │   └── termination_cfg.py
    │   └── mdp/
    │       ├── __init__.py
    │       ├── observation.py
    │       └── reward.py
    └── snow_ball/
        ├── __init__.py
        ├── snow_ball_env_cfg.py
        ├── README.md
        ├── agents/
        │   ├── __init__.py
        │   ├── rsl_rl_ppo_cfg.py
        │   └── compositional_rsl_rl_cfg.py
        ├── env_cfg/
        │   ├── __init__.py
        │   ├── action_cfg.py
        │   ├── commands_cfg.py
        │   ├── curriculum_cfg.py
        │   ├── event_cfg.py
        │   ├── observation_cfg.py
        │   ├── physics_cfg.py
        │   ├── reward_cfg.py
        │   ├── scene_cfg.py
        │   └── termination_cfg.py
        └── mdp/
            ├── __init__.py
            ├── observation.py
            └── reward.py
```
