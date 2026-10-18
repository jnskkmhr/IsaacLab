## Wandb login
```bash
uv run --with wandb wandb login
```

## Training
```bash
isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Tree --num_envs 32 --viz none
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Tree --num_envs 16 --viz none
```

## Inference
```bash
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Tree-Play --num_envs 4 --viz newton_gl --wandb_run 4blqnxbi
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Tree-Play --num_envs 4 --viz newton_rtx --wandb_run 4blqnxbi
```