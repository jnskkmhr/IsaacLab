# WorldParticle (`contrib/legacy/particle_world`)

Pure **NVIDIA Warp + PyTorch** adaptation of MIT CSAIL's **WorldParticle** architecture
([*WorldParticle: Unified World Simulation of Lagrangian Particle Dynamics via Transformer*](https://worldparticle.github.io/WorldParticle_Web/),
[`WorldParticle/WorldParticle`](https://github.com/WorldParticle/WorldParticle)) for granular sand and rigid–particle interaction in Isaac Lab.

---

## Architecture Overview

Each physics step ($\Delta t = 0.005\text{ s}$, 200 Hz) advances $N$ Lagrangian sand particles in four stages:

1. **Explicit Predictor & Physics-Informed Lookahead (`network/cconv_warp.py:integrate_pos_vel`)**:
   - Advances particle positions and velocities to the lookahead predictor state $(\mathbf{x}_{\text{pred}}, \mathbf{v}_{\text{pred}})$ and evaluates analytical signed distances, outward normals, and relative surface velocities against all rigid colliders and tray boundaries in Warp (`particle_features_kernel`, `collider_features_kernel`).
2. **Stage 1 — Warp Continuous 3D Convolution Tokenizer (`network/cconv_warp.py:ContinuousConvWarp`, `network/model.py:LocalFeatureExtractor`)**:
   - Replaces Open3D's `FixedRadiusSearch` + `ops.continuous_conv` with allocation-free NVIDIA Warp `wp.HashGrid` kernels (`cconv_fluid_basis_kernel` and `cconv_obstacle_basis_kernel`).
   - Maps neighbor offsets $\Delta \mathbf{x}_{ij} / R$ inside the support ball ($R = 0.072\text{ m}$) to the $[-1, 1]^3$ cube via concentric ray stretching (`ball_to_cube_radial`), weights them by the cubic `poly6` window $(1 - (r/R)^2)^3$, and accumulates trilinearly interpolated `[4, 4, 4]` filter bases for both **fluid/sand neighbors (`conv0_fluid`)** and **analytical collider surfaces (`conv0_obstacle`)**, alongside local neighbor packing density (`packing`).
3. **Stage 2 — 3D-RoPE Hierarchical Super-Particle Encoder (`network/merge.py`, `network/attention.py`, `network/model.py:SuperParticleEncoder`)**:
   - Applies iterative **$k$-Partite Token Merging (ToMe)** (`k_partite_soft_matching`) with **3D Rotary Position Embeddings (`ParticleRotaryEmbedding`)**, **QK-RMSNorm**, and **SwiGLU** feed-forward blocks (`MultiHeadAttentionWithTOME`).
   - Tracks 3D super-particle coordinates $\mathbf{x}_{\text{super}}$ in $O(N)$ memory via token-size-weighted averaging (`merge_wavg`), avoiding the $O(N^2)$ dense source-matrix allocation of upstream `merge_source`.
4. **Stage 3 — 3D-RoPE Cross-Attention Decoder & `velhead` Output (`network/model.py:SuperParticleDecoder`)**:
   - All $N$ query particles at $\mathbf{x}_{\text{pred}}$ cross-attend to the $N_s$ super-particles at $\mathbf{x}_{\text{super}}$ using PyTorch's native FlashAttention-2 / cuDNN fused `F.scaled_dot_product_attention` with 3D RoPE on both queries and keys.
   - Skip-connects the fine-scale Continuous Convolution features into the output MLP (`velhead=True`) to predict per-particle position and velocity corrections $(\Delta \mathbf{x}_{\text{corr}}, \Delta \mathbf{v}_{\text{corr}})$ plus per-particle inertial and collider advection gates.

---

## Layout

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/legacy/particle_world/
├── __init__.py
├── README.md
├── solver_cfg.py          # ParticleWorldSolverCfg for Newton coupled simulation
├── manager.py             # NewtonParticleWorldManager & NewtonParticleWorldCouplerManager
├── solver.py              # ParticleWorldSolver (WorldParticleNetwork + ProxyEstimator)
├── network/
│   ├── __init__.py
│   ├── cconv_warp.py      # Warp HashGrid [4,4,4] ContinuousConv + differentiable collider features
│   ├── rope.py            # Continuous 3D ParticleRotaryEmbedding & freqs_to_cos_sin
│   ├── merge.py           # Zero-sync k_partite_soft_matching & O(N) merge_wavg position tracking
│   ├── attention.py       # RMSNorm, FeedForwardSwiGLU, MultiHeadAttention (Flash SDPA), MultiHeadAttentionWithTOME
│   └── model.py           # LocalFeatureExtractor, SuperParticleEncoder, SuperParticleDecoder, WorldParticleNetwork
└── showcase/
    ├── __init__.py
    ├── contact_forces.py  # 200 Hz material-on-object contact force reconstruction in Warp
    └── run_showcase.py    # Train, evaluate 6.0 s open-loop rollouts, plot, and render 50 fps 3D videos
```

---

## Running the Benchmark Showcase

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/legacy/particle_world/showcase/run_showcase.py \
  --cases horizontal-flat vertical-flat circle-flat rolling-flat \
  --steps 800 \
  --output video/benchmarks/particle_world_showcase
```
