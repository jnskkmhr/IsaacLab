## Train 
```bash
uv run --with wandb isaaclab train \
    --rl_library rsl_rl \
    --task IsaacContrib-WBC-G1-29dof \
    --num_envs 8192 \
    --viz newton_gl \
    --video
```