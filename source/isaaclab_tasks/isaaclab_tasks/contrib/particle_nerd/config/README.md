# Interactive box and snow-ball examples

These tasks inherit the current scenes in `contrib/particle_world_batch/config/box`
and `contrib/particle_world_batch/config/snow_ball`. They retain the material,
particle initialization and jitter, object mass and geometry, tray, initial
velocities, actions, observations, rewards, resets, and viewer settings. Their
learned particle solver is Particle NeRD's object Transolver.

Run from the Isaac Lab repository root:

```bash
uv run isaaclab play --task IsaacContrib-ParticleNeRD-Box-v0 \
  --num_envs 1 --viz newton_gl

uv run isaaclab play --task IsaacContrib-ParticleNeRD-SnowBall-v0 \
  --num_envs 1 --viz newton_gl
```

No RL policy checkpoint is needed. Playback applies zero action; the rigid body
moves under gravity, reconstructed particle contact forces, and rigid tray
contacts. Right-click and drag the rigid object to apply a mouse force in the
OpenGL viewer. The reference six-second episode timeout and reset are preserved.
For a longer interaction, append `env.episode_length_s=60` to either command.
Use `--num_envs 4` to view the inherited four-environment setup.

For the existing WorldParticle counterparts, use the same zero-action runner:

```bash
uv run python scripts/environments/zero_agent.py \
  --task IsaacContrib-ParticleWorld-Box-v0 --num_envs 1 --viz newton_gl

uv run python scripts/environments/zero_agent.py \
  --task IsaacContrib-ParticleWorld-SnowBall-v0 --num_envs 1 --viz newton_gl
```

The corresponding native MPM task IDs are
`IsaacContrib-ParticleWorld-Box-MPM-v0` and
`IsaacContrib-ParticleWorld-SnowBall-MPM-v0`; use them with the same zero-agent command.

## Preserved scene settings

| Setting | Box | Snow ball |
|---|---|---|
| Physics interval | 0.005 s | 0.005 s |
| Rigid substeps per coupling step | 4 | 2 |
| Initial particle bed dimensions | 2.0 × 1.2 × 0.3 m | 2.0 × 1.2 × 0.3 m, with the reference slope profile |
| Particle-generation voxel size | 0.02 m | 0.04 m |
| Particles per cell per axis | 1.25 | 1.25 |
| Material density | 2,500 kg/m³ | 400 kg/m³ |
| Young's modulus | 10¹⁵ Pa | 1.4 × 10⁶ Pa |
| Rigid object | 0.24 × 0.18 × 0.12 m box, 75 kg | Radius 0.10 m sphere, 4 kg |
| Initial object position in its environment | (0, 0, 0.35) m | (−0.65, 0, 0.22) m |
| Initial linear velocity | (0, 0, 0) m/s | (0.35, 0, 0) m/s |
| Initial angular velocity | (0, 0, 0) rad/s | (0, 3.5, 0) rad/s |
| Contact normal / tangential mass factors | 100 / 40 | 100 / 40 |
| Contact iterations / relaxation | 200 / 0.02 | 200 / 0.02 |

The scene configuration remains the source of truth. The snow surface is
`z(x) = −0.12 x + 0.015 sin(2πx / 1 m)`, with the reference spawn clipping.
The NeRD feature encoder reads the actual tray geometry and material values.
Each environment's coordinates are translated to its own tray origin before
network inference.

## Checkpoint and inference method

The default checkpoint path is `../checkpoint/object_topk.pt`, configured in
`../object_solver_cfg.py`. This file must come from the new object retraining
run with schema `object_particle_nerd_topk_v2`. The historical `object.pt`
checkpoint was trained with raw full-particle predictions; it is rejected by
this solver and must not be renamed to bypass the check. Until the new run
finishes and its checkpoint is installed, the default play command cannot load
its model.

The object adapter uses the current G1 Particle NeRD method: automatic Top-K
selection, 512 background samples, selection refresh every four steps, the
same blend, free-surface ballistics, Argon's speed-dependent displacement
stabilization, and tray-floor/wall projection. Only primitive collider geometry
changes. Training and evaluation call the same shared update as interactive
playback. There are zero native MPM warm-up steps and no force filtering.

The force estimator is the G1 proxy estimator, with normal/tangent/horizontal/
upward mass scales 100/40/8/2, normal exponent 3, and its body-velocity solve.
The neural network receives pre-rigid body pose and velocity; contact
reconstruction receives the separate rewound proxy free velocity. CUDA graph
capture remains disabled for these interactive object tasks.

The retraining dataset covers flat dry sand and box/sphere/cylinder interactions.
It does not include the snow material or the box example's different density
and particle spacing. The scene configurations retain those requested material
settings; prediction accuracy there must be evaluated separately.

## Validation

```bash
uv run python -m pytest source/isaaclab_tasks/test/contrib/test_particle_nerd_objects.py -q
```

The checks cover reference scene parity, G1 method-parameter parity, translated
environments, material reactions, independent resets, and failure if native MPM
is constructed or stepped. Integration tests use a synthetic network and do not
establish trained prediction accuracy. Previously recorded raw-model videos do
not represent this corrected method.
