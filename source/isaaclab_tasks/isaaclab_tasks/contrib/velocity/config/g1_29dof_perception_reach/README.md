# G1 perception reach

## Train

```bash
# AME
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-AME --num_envs 4096 --viz newton_gl --video

# CNN
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-CNN --num_envs 4096 --viz newton_gl --video
```

## Play

```bash
# AME
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-AME-Play \
  --num_envs 1 --wandb_run xxx --viz newton_gl

# CNN
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-CNN-Play \
  --num_envs 1 --wandb_run xxx --viz newton_gl
```
