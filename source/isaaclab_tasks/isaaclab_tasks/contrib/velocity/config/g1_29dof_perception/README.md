# G1 perception

## Train

```bash
# AME
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-AME --num_envs 4096 --viz newton_gl --video

# CNN
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-CNN --num_envs 4096 --viz newton_gl --video
```

## Play

```bash
# AME
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-AME-Play \
  --num_envs 4 --checkpoint /path/to/ame/model.pt --viz newton_gl

# CNN
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-CNN-Play \
  --num_envs 4 --checkpoint /path/to/cnn/model.pt --viz newton_gl
```
