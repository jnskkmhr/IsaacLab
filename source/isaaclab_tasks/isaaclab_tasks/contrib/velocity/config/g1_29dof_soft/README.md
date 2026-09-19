## Training
```bash
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft --num_envs 4096 --viz newton_gl --video
```

## Inference
```bash
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft-Play --num_envs 4 --viz newton_gl --wandb_run 4blqnxbi
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-G1-29dof-Soft-Play --num_envs 4 --viz newton_rtx --wandb_run 4blqnxbi
```


## Symmetry augmentation

The registered PPO runner enables left/right data augmentation using the shared
[term-driven mirror implementation](../vel_mdp/README.md). Policy, critic,
privileged, and logging observation terms all declare mirror functions. Joint
signals swap left/right channels with the appropriate signs; quaternions use
XYZW order. Hybrid contact observations swap feet and reflect force vectors,
while terrain material scalars use an identity mirror.

`physics_callback` computes soft-contact forces during simulation but has **zero
policy action dimensions**. `MirrorPhysicsCallbackActionCfg` gives its empty
action slice `mirror_identity`; augmentation does not execute or modify the
contact solver. Its derived contact observations are mirrored separately because
they are visible to the critic.

Hybrid force observations now threshold the magnitude of each force component,
retaining both positive and negative components before signed-log compression.
This is necessary for reflected observations to match observations computed from
reflected forces. Observation/action dimensions are unchanged, but resuming an
existing run uses corrected force inputs and enables augmented training samples.
Set `algorithm.symmetry_cfg=None` in the runner config to disable augmentation.
