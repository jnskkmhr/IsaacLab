# G1 object pushing with unconfined MPM sand

This task uses the current whole-body object-push controller and a **3 × 2 × 0.3 m**
grid of native implicit-MPM sand on a large flat ground plane. The robot has closed, non-policy-controlled
fingers and pushes a **5 kg object**. Each environment independently samples one of
three shapes with equal probability during scene construction: a 0.65 m cube,
a sphere of radius 0.325 m, or an upright cylinder of radius 0.325 m and height
0.65 m. The shape stays fixed across episode resets. All three use convex-mesh
colliders because this MJWarp version requires matching collider types across worlds.
The shared scene asset name is `object`, with prim path `{ENV_REGEX_NS}/Object`.
Both feet and the object are coupled to MPM; MJWarp resolves robot–object contact.

Hand targets follow each shape's surface and are projected into the existing
reachable wrist-pose dataset. Sphere targets stay aligned with world up while
the sphere rolls; only boxes and cylinders retain the upright-object termination.
Student observation dimensions and ordering remain unchanged. Command, reward,
and observation names now use `object` (for example `commands.object_velocity`).

All particles in each environment remain active throughout an episode.
There is no moving patch, recycling, or cached/frozen background. Displaced sand
can accumulate around the object and remains deformed after it passes. Episode resets
restore the selected environment's bed; other environments retain their particles.
There are no retaining walls or resets at the original sand footprint. The robot
and object can leave the sand and continue onto the rigid ground. Sand can spread
beyond its initial footprint; it is neither clipped nor recycled.

## Run

From `/home/jkamohara/isaac/IsaacLab`:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-MPM-ObjectPush \
  --num_envs 4 --viz newton_gl --logger wandb
```

The default agent uses compositional PPO with locomotion teacher
`hpmmpox5/model_5500.pt` and closed-hand manipulation teacher `6pbvl2df/model_9500.pt`.
The locomotion teacher receives 790 proprioceptive history values plus eight current
command values (velocity, integrated world heading quaternion, and height).
Leg/waist gains and joint offsets match that teacher; arm gains match the manipulation
teacher. Their paths are in
`agents/compositional_rsl_rl_cfg.py`. The W&B project is `g1_29dof_mpm_object_push`.
The actor still controls all 29 body joints; this is not the lower-body-only task.

```bash
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-MPM-ObjectPush-Play \
  --num_envs 1 --viz newton_rtx \
  --wandb_run xxx 'env.video_recorders=[]'
```

The observation/action layout matches the current rigid object-push task, allowing
its student checkpoints to be evaluated on sand. This does not establish policy
success on sand. The train command samples forward speeds of 0.1–0.4 m/s and
includes stop commands; Play commands 0.4 m/s forward.

## Configuration

- `env_cfg/scene_cfg.py`: per-environment sand grid, sand material, voxel size,
  object shape set, and robot PD gains. A shared 1,000 × 1,000 m ground plane lies
  at z=−0.3 m; the initial sand surface remains local z=0. The unconfined grid
  settles and spreads at its edges. Existing height commands/rewards still use
  the initial sand-surface reference, not the lower exposed ground.
- `env_cfg/physics_cfg.py`: rigid and MPM entries; feet and object share the proxy
  mapping. MJWarp owns the ground collider, and the MPM view also includes it
  as a static collider. Sparse capacities scale with the final environment count.
- `env_cfg/event_cfg.py`: reset the robot near the rear of the bed and place the
  object 0.65–0.8 m ahead of it. Material parameters are sampled independently
  per environment at every reset, in both Train and Play:
  friction 0.3–0.9 and density 1,000–3,000 kg/m³ (uniform), Poisson ratio 0.3,
  Young's modulus 10⁷–10⁹ Pa and yield pressure 10⁸–10⁹ Pa (log-uniform).
  Use `env.events.mpm_material=null env.events.mpm_material_log=null` to retain
  the nominal material from `scene_cfg.py`.
- `env_cfg/reward_cfg.py`: current push rewards, including zero-command standing.
  Gait terms use combined MPM/rigid foot forces. Foot pitch is measured against
  the initial horizontal surface, not the evolving pile's local slope.
- `object_push_env.py`: task-local extraction of object-filtered MJWarp forces.
  Actual palm/finger contact earns hand-contact credit; sand support and knee
  contact do not. The standard contact sensor is unsupported by the coupled backend.

## Sand rendering

The default viewers are built-in `NewtonGLVisualizerCfg` and
`NewtonRTXVisualizerCfg`, following `robot` with `origin_type="asset"`.
Training records Newton GL video every 2,000 steps for 200 frames, with particles
hidden. Play enables normal particle visualization. Hidden particles remain
fully simulated. Optional surface reconstruction helpers remain in `util/visualizer.py`;
they are not enabled by the default viewer configuration.

## Verification

```bash
uv run --no-sync python -m pytest source/isaaclab_tasks/test/contrib/test_g1_fixed_mpm_push.py -q
```

The two-environment GPU test checks actual hand–box forces, a prescribed box sweep
that lifts and spreads sand, ground support in both worlds, fixed particle count,
and independent resets. It also checks foot support on rigid ground outside the
initial sand footprint and that leaving that footprint does not reset an episode. Only the test makes the box kinematic to impose a repeatable sweep; the
training/play box remains dynamic. Two local teacher-guided training updates were
also checked with the default dynamic box. These checks do not measure trained
pushing performance.

## Standing and contact rewards

Zero commands earn positive Gaussian rewards for low root velocity (widths 0.2 m/s
and 0.5 rad/s) and remaining at the reset or stop-entry XY position (width 0.15 m).
Both weights are +5. Hand approach/contact and box-tracking credit are disabled
while stopped. Foot spacing penalizes only signed lateral separation below 15 cm.
Box tracking during movement requires actual hand-box contact from MJWarp; sand
support forces cannot earn that credit. All bed particles remain simulated.
