## Moviepy install 
Need this for video logging
```bash 
pip install 'moviepy<2'
```
## Wandb login
```bash
uv run --with wandb wandb login
```

## Training
```bash
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid --num_envs 4096 --viz newton_gl --video
```

## Inference
```bash
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Play --num_envs 4 --viz newton_gl --wandb_run 4blqnxbi
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Play --num_envs 4 --viz newton_rtx --wandb_run 4blqnxbi
```