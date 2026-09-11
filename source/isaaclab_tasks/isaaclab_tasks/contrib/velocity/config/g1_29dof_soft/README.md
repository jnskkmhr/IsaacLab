## Training
```bash
isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft --num_envs 4096 --viz none
```

## Inference
```bash
isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft-Play --num_envs 100 --viz newton_gl --wandb_run xxx
```
