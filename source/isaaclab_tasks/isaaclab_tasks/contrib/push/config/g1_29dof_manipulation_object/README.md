# Pinned G1 manipulation-object test

Evaluate the existing reaching policy without retraining. The legless robot has a fixed pelvis,
zero waist position targets, and closed fingers. Only the 14 arm joints receive policy actions.
The 76-dimensional actor observation, default joint offsets, action scale, and PD gains match
the manipulation teacher `6pbvl2df/model_19999.pt`. Object state is not added to the actor input.

## Run

From `/home/jkamohara/isaac/IsaacLab`:

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Manipulation-Object-Play \
  --num_envs 1 --viz newton_rtx \
  --wandb_run 6pbvl2df --wandb_project g1_29dof_manipulation \
  "env.video_recorders=[]"
```

To use a local checkpoint, replace the two W&B arguments with `--checkpoint /absolute/path/model_19999.pt`.

## Test geometry and command

- A 5 kg cuboid, 0.24 m long × 0.44 m wide × 0.20 m tall, rests on a fixed platform.
- Platform top: world z=0.95 m. Initial object rear face: world x=0.32 m.
- Torso link starts at world z=1.0 m. The USD contains a 0.837 m offset, accounted for in its spawn position.
- Wrist targets are in `torso_link`: x moves from 0.08 to 0.32 m, y stays at ±0.18 m, z stays at 0.05 m.
- Both target quaternions are identity XYZW. Hold the starting target for 2 s, advance smoothly for 4 s,
  then hold the endpoint. The episode resets after 10 s.
- The hand extends forward from the wrist, so object contact precedes the wrist reaching the rear face.
- Reset arms to a collision-checked near-torso pose; preserve the teacher's default joint offsets.
- Built-in pose markers show commanded and measured wrist frames. `object_contacts` measures robot contacts
  specifically against the object, including finger links.

This is a deterministic forward-stroke diagnostic, not an object-aware policy or a training task.
There is no workspace projection, object-following target, force disturbance, or loss curriculum.
A pinned pelvis absorbs reaction forces: success here does not establish full-body pushing stability.

The scene, event, command, observation, and physics configurations are standalone under `env_cfg/`.
Reusable MDP code is imported from the existing manipulation task. Rewards and curricula are empty
because this task evaluates a frozen checkpoint.
