## Agent configuration

`agents/compositional_ppo_cfg.py` configures compositional PPO for the flat and rough tasks,
with the existing observation order, networks, PPO settings, and symmetry augmentation.
Matching ordinary RSL-RL checkpoints with separate MLP actor/critic weights can be loaded directly.
The LSTM config disables symmetry augmentation, which RSL-RL does not support for recurrent
policies. Ordinary RSL-RL recurrent checkpoints require conversion
and are not supported by the MLP compatibility loader.

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
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Rough-G1-29dof-Rigid --num_envs 4096 --viz newton_gl --video
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid --num_envs 4096 --viz newton_gl --video
```

## Inference
```bash
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Play --num_envs 4 --viz newton_gl --wandb_run 4blqnxbi
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Play --num_envs 4 --viz newton_rtx --wandb_run 4blqnxbi
```