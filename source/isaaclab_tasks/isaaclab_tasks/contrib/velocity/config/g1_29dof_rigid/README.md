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

uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Rough-G1-29dof-Rigid --num_envs 4096 --viz newton_gl --video --agent rsl_rl_fast_sac_cfg_entry_point
```

## Inference
```bash
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Play --num_envs 4 --viz newton_gl --wandb_run 4blqnxbi
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Play --num_envs 4 --viz newton_rtx --wandb_run 4blqnxbi
```

## FastSAC gait reward experiment

The default rewards remain unchanged. Select `env.rewards=fast_sac` to enable stronger
hip-yaw and knee-posture regularization, foot-yaw alignment, swing/stance-duration
balance, and a smaller, narrower foot-clearance reward. Action-rate penalties retain
the current lower/upper-body weights of -0.1/-0.25 and second-difference weight -0.025.
They measure policy actions, so their physical effect depends on the action scale.

```bash
uv run --with wandb isaaclab train \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Rigid \
  --num_envs 4096 --viz newton_gl --video --video_length 200 \
  --agent rsl_rl_fast_sac_cfg_entry_point \
  env.rewards=fast_sac \
  env.actions=joint_range_action env.actions.joint_pos.scale=1.0 \
  agent.run_name=gait_reward_v1 agent.replay_capacity=500000
```

The 500,000-transition replay capacity fits the local 32 GB GPU; the comparison run
used 1,000,000 transitions on a larger GPU. The explicit action scale matches q5egtbab. Keep the same action
preset and scale when playing the resulting checkpoint. This is an experimental
reward setting; a higher total reward alone does not establish a better gait.
