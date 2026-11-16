A. Batched Environment Rollout (zero_agent.py / random_agent.py)

# Headless batched rollout (e.g. 4 parallel environments)

```bash
uv run python scripts/environments/zero_agent.py \
    --task IsaacContrib-ParticleNeRDLocal-Box-v0 \
    --num_envs 4


uv run python scripts/environments/random_agent.py \
    --task IsaacContrib-ParticleNeRDLocal-Go2-v0 \
    --num_envs 4 \
    --visualizer newton_gl
```
Swap --task with any of the 6 tasks:

```bash
--task IsaacContrib-ParticleNeRDLocal-Box-v0
--task IsaacContrib-ParticleNeRDLocal-Sphere-v0
--task IsaacContrib-ParticleNeRDLocal-Cylinder-v0
--task IsaacContrib-ParticleNeRDLocal-Franka-v0
--task IsaacContrib-ParticleNeRDLocal-Go2-v0
--task IsaacContrib-ParticleNeRDLocal-G1-v0
```


B. RSL-RL PPO Training (train.py)

# Train with RSL-RL PPO across N batched environments (headless)

```bash
uv run python scripts/reinforcement_learning/rsl_rl/train.py \
    --task IsaacContrib-ParticleNeRDLocal-Go2-v0 \
    --num_envs 16
# Or using ./isaaclab.sh -p wrapper:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task IsaacContrib-ParticleNeRDLocal-G1-v0 \
    --num_envs 16
```

C. RSL-RL Policy Play / Evaluation (play.py)

# Play a trained RSL-RL policy with the live Newton OpenGL viewer
```bash
uv run --with wandb --extra ovrtx isaaclab play \
    --rl_library rsl_rl \
    --task IsaacContrib-ParticleNeRDLocal-G1-v0 \
    --num_envs 4 \
    --visualizer newton_gl
```
# Play and record a video Headless (or with --checkpoint /path/to/model.pt)
```bash
uv run --with wandb --extra ovrtx isaaclab play \
    --rl_library rsl_rl \
    --task IsaacContrib-ParticleNeRDLocal-G1-v0 \
    --num_envs 4 \
    --wandb_run jqur5jhm \
    --wandb_project g1_29dof_mpm_mixed_terrain \
    --viz newton_rtx \
    "env.video_recorders=[]"
```