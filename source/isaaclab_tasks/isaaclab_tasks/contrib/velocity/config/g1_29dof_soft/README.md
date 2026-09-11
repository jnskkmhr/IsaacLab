## Dependencies
UV does not resolve dependency for offline video recording.
```bash
pip install 'moviepy<2'
```

## Training 
```bash
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft --num_envs 4096 --viz newton_gl --run_name xxx
```

## Inference 
```bash 
# newton gl visualizer
uv run --with wandb --extra isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft-Play --num_envs 50 --viz newton_gl --wandb_run 4blqnxbi
# newton rtx visualizer
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft-Play --num_envs 50 --viz newton_rtx --wandb_run 4blqnxbi
```