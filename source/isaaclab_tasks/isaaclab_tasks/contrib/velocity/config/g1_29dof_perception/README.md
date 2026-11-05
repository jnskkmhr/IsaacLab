# G1 perception

## Encoder loading and freezing

The existing AME and CNN configurations remain the defaults. Each task, including
its Play variant, additionally exposes these `--agent` selections:

| Agent entry | Encoder behavior |
| --- | --- |
| `rsl_rl_encoder_train_cfg_entry_point` | Existing fresh initialization; train normally |
| `rsl_rl_encoder_pretrained_cfg_entry_point` | Load a compatible complete encoder term and freeze parameters and normalization state |
| `rsl_rl_encoder_finetune_cfg_entry_point` | Load the same complete encoder term and train normally |

The selected task determines whether these keys resolve to CNN or AME variants.
Both preserve the corresponding base configuration's network architecture, inputs,
observations, terrain, rewards and PPO settings. The actor and critic train in all
three modes. `run_name` distinguishes loaded variants in logs.

For either loaded variant, provide
`agent.networks.map_encoder.pretrained_cfg.path=/absolute/path/model.pt`.
The `???` default requires an explicit path through Hydra. The default checkpoint
selector uses `network_state_dict`, prefix `terms.map_encoder.`, and strict loading.
Adjust selectors only after inspecting the checkpoint. Architecture, input dimensions,
preprocessing and normalization must match; CNN and AME checkpoints are not interchangeable.
No pretrained model is selected or bundled. `--checkpoint` loads a full runner and
is a separate operation.

Variants declare independently copied `networks` defaults following the companion's
examples, preserving constructor and `replace(networks=...)` overrides. Use the current
`compositional_rsl_rl` package (validated at companion revision `2ff331a`). These options
do not launch a sweep, add sensors or connect reference-motion data to training.

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
