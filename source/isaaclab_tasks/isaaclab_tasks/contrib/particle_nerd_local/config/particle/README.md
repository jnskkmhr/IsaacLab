# ParticleNeRD-Local Particle Task

Passive manager-based scene for a free block of sand (`0.40 m` cube, sphere or vertical cylinder, about `1.1k–1.7k` particles) dropped or tossed onto open flat ground. Nothing is controlled; the scene exists to compare `ParticleNeRDLocalSolver` with implicit MPM on the same initial state and to record both in the viewer.

The sand is released `0.20 m` above the floor, centred at `x = -0.15 m`. The solver expects one floor and four walls, so the four walls sit 10 m away (hidden) and a hidden kinematic `anchor` body 10 m above the ground fills the rigid collider slot the model was trained with. This is the layout of the free-fall training data.

## Registered Gym Environments

`IsaacContrib-ParticleNeRDLocal-Particle-<Shape>-<Motion>-v0` (Particle NeRD Local) and
`IsaacContrib-ParticleNeRDLocal-Particle-<Shape>-<Motion>-MPM-v0` (implicit MPM ground truth), where
`<Shape>` is `Cube`, `Sphere` or `Cylinder` and `<Motion>` is:

- `Drop`: released from rest.
- `Throw`: released at `1.0 m/s` along `+x` and `0.25 m/s` up.

Episodes last 6 s at 200 Hz (`decimation = 1`).

## Usage

```bash
uv run --no-sync python scripts/environments/zero_agent.py \
    --task IsaacContrib-ParticleNeRDLocal-Particle-Cube-Throw-v0 \
    --num_envs 1 \
    --visualizer newton_gl
```

Add `--video viz:newton_gl --video_length 1200` to record the viewer.

There is no `agents/` package: the scene has no learning task.
