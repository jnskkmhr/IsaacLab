# Agent edit log

Record of the changes an agent made to this task package, with the measurements behind them.
This is a scratch/working document — per `AGENTS.md` it should not be committed.

## Files touched

The whole package is still untracked, so `git diff` shows nothing; this is the authoritative list.

| File | What changed | Section |
| --- | --- | --- |
| `g1_mpm_env.py` | Rebuilt around the coupler proxy feedback wrench; added a `step()` override that scrubs NaN/±inf from the reward buffer | 1, 3 |
| `g1_mpm_env_cfg.py` | Dropped the dead sinkage/height-scan fields, added `foot_contact_force_threshold = 5.0` | 1 |
| `mdp/observations.py` | Added the six privileged terms, removed the three invented ones | 1 |
| `mdp/terminations.py` | New `root_state_not_finite` | 3 |
| `mdp/__init__.pyi` | Exported `root_state_not_finite` | 3 |
| `env_cfg/observation_cfg.py` | `PrivilegedObsCfg` / `LoggingObsCfg` realigned with `g1_29dof_soft` | 1 |
| `env_cfg/termination_cfg.py` | New `solver_diverged` term | 3 |
| `env_cfg/physics_cfg.py` | `grid_padding` 50 → 0; `critical_fraction` 0.5 → 0.0 | 2a, 2c |
| `env_cfg/scene_cfg.py` | New `FOOT_CONTACT_MARGIN = 0.75 * MPM_VOXEL_SIZE`, applied to the robot spawn through `NewtonCollisionPropertiesCfg(contact_margin=..., contact_gap=0.0)` | 2b, 3 |
| `README.md` | Tuning notes for the three hard constraints | 4 |

Constants as shipped: `MPM_VOXEL_SIZE = 0.05`, `MPM_COLLIDER_MARGIN = 0.5 * MPM_VOXEL_SIZE`
(bed boxes, unchanged), `FOOT_CONTACT_MARGIN = 0.75 * MPM_VOXEL_SIZE` (robot, new).

`MPM_VISUAL_COLOR` was changed to `(0.32, 0.24, 0.21)` by the user, not by the agent.

---

## 1. Privileged observation group aligned with `g1_29dof_soft`

**Problem.** `PrivilegedObsCfg` carried two invented terms (`foot_sinkage`, `sand_height_scan`)
and was missing `foot_contact_force` and `terrain_material_parameters`.

The premise behind those terms was also wrong: an earlier note in `mdp/rewards.py` and `README.md`
claimed "the MPM solver exposes no per-collider impulse to Isaac Lab". It does. Because the feet
are proxies of the rigid entry inside the MPM entry, the granular reaction on each foot is exactly
the proxy feedback wrench the coupler hands back, held in
`NewtonManager._solver._proxy_mappings[i].coupling_forces` (a `wp.spatial_vector` per Newton body;
components `[0:3]` are the linear force in N).

**Changes.**

| File | Change |
| --- | --- |
| `g1_mpm_env.py` | Rewritten around the real proxy feedback wrench. Resolves the foot bodies to Newton body ids by prim-path label, harvests `coupling_forces` once per step, and derives `foot_contact_force`, `foot_contact`, `foot_first_contact`, `foot_air_time`, `foot_contact_time` from it. |
| `mdp/observations.py` | Added `foot_contact`, `foot_contact_force` (clamped ±1000 N, thresholded, log-compressed), `foot_contact_force_raw`, `foot_air_time`, `terrain_material_parameters` (zero placeholder). Removed `foot_sinkage`, `foot_sand_surface_height`, `sand_height_scan`. |
| `env_cfg/observation_cfg.py` | `PrivilegedObsCfg` = `base_lin_vel`, `foot_height`, `foot_contact`, `foot_contact_force`, `foot_air_time`, `terrain_material_parameters`. `LoggingObsCfg.contact_forces` now uses `foot_contact_force_raw`. |
| `g1_mpm_env_cfg.py` | Replaced the dead sinkage/scan fields with `foot_contact_force_threshold: float = 5.0`. |

**Verified in-container.** Privileged group = 16 floats/frame × history 10 = 160. Term dims:
`base_lin_vel (30,)`, `foot_height (20,)`, `foot_contact (20,)`, `foot_contact_force (60,)`,
`foot_air_time (20,)`, `terrain_material_parameters (10,)`. Forces read 40–113 N with sensible
contact flags and air times.

**Note.** `terrain_material_parameters` returns zeros by design — there is no single per-foot
stiffness in an MPM bed; it exists only to keep the observation layout identical to
`g1_29dof_soft`.

---

## 2. MPM solver settings derived from the standalone Newton example

Two separate defects: the simulation ran at ~0.4 fps, and the robot sank to the bottom of the bed.

### 2a. `grid_padding: 50 → 0` (`env_cfg/physics_cfg.py`) — the speed fix

The standalone example pads by 50 voxels, but it runs `grid_type="fixed"`. On a **sparse** grid,
`_sparse_rebuildable` requires `grid_padding == 0`; with padding the grid is not rebuildable, which
refuses CUDA-graph capture and reallocates a grid padded by 50 × 0.05 m = 2.5 m in every direction
on every step. It also hard-crashes at ≥ 4 envs with
`RuntimeError: Keys and values array storage must be large enough to contain 2*count elements`
(from `_compute_coloring` → `wp.utils.radix_sort_pairs`).

**Result:** 0.4 fps → 178 ms/step at 4 envs, 95–110 ms/step at 1 env.

Padding is only transferable together with `grid_type="fixed"`.

### 2b. Foot collider margin (`env_cfg/scene_cfg.py`) — the main sinking cause

The MPM collider thickness comes from the Newton shape's contact margin. The bed boxes had
`MPM_COLLIDER_MARGIN = 0.025`; the proxied ankle links inherited `0`, so each sole was a
zero-thickness shell that occupied essentially no 0.05 m voxel and the bed carried no load.
Confirmed from the solver: `collider_max_thickness = [0, 0, 0.025, 0.025, ...]` (indices 0–1 are
the feet).

Added `FOOT_CONTACT_MARGIN = MPM_VOXEL_SIZE` and applied it to the robot spawn via
`NewtonCollisionPropertiesCfg(contact_margin=FOOT_CONTACT_MARGIN, contact_gap=0.0)`.

Measured foot height at step 20 (nominal standing ankle height ≈ 0.048 m), 1 env, zero actions:

| foot margin | foot_z | reaction per foot |
| --- | --- | --- |
| 0 (before) | −0.20 (bed bottom) | 0–95 N |
| 0.01 | −0.205 | 0–80 N |
| 0.025 | −0.09 | ~200 N |
| 0.0375 | −0.048 | 300–340 N |
| 0.05 | +0.001 | 150–350 N |

`FOOT_CONTACT_MARGIN` was subsequently set to `0.75 * MPM_VOXEL_SIZE` (0.0375) rather than a full
voxel — see section 3, where the full-voxel margin turned out to destabilise the solve.

Refining the voxel is **not** a substitute: at `voxel 0.025 / margin 0.025` the robot still sank
14 cm and the step cost doubled (215 ms/step). Absolute margin is what matters, so
`MPM_VOXEL_SIZE` stays at 0.05.

### 2c. `critical_fraction: 0.5 → 0.0` (`env_cfg/physics_cfg.py`) — second sinking cause

`critical_fraction` is the voxel fill fraction below which the yield surface collapses. The bed is
sampled at 0.04 spacing against a 0.05 voxel, so a voxel holds 1 or 2 particles and the fill
fraction oscillates between 0.46 and 0.91 — straddling 0.5. Roughly half the bed therefore loses
its shear strength and behaves as a frictionless fluid. The standalone G1 example can use 0.5
because its sampling fills every cell; `example_mpm_anymal.py` uses 0.0, which is also the Newton
default.

Effect was masked by 2b. With both fixed, foot_z holds at 0.033–0.06 instead of 0.001.

### Things tested and rejected

- Per-particle material **does** reach the sand — `mpm:friction = 0.577`, `young_modulus = 15 MPa`,
  `poisson_ratio = 0.3`, `yield_pressure = 1e12` across all 39,375 particles. (Correct accessor is
  the `model.mpm` namespace, not `get_custom_attribute("mpm:friction")`.)
- Reference `max_iterations = 50` / `tolerance = 1e-6`: no improvement, 12 % slower. Kept 25 / 1e-5.
- No effect on sinking from `mass_scale` (1 / 26.6 / 200), proxy `mode` lagged vs staggered,
  coupler `iterations`, `num_substeps`, `young_modulus = 1e15`, `yield_pressure = 1e15`, bed depth.

### Validation of 2a–2c

4 envs, 60 steps, zero actions: **174 ms/step (≈ 5.75 fps, up from 0.4)**, no NaN, feet held at
0.00–0.06 m, 150–300 N per foot summing to body weight, bed piling up (`sand_z_max ≈ 0.05`).

---

## 3. Per-world NaN blow-up ("an environment disappears and comes back")

**Diagnosis.** At some step a whole world's state — all 39,375 particles *and* the robot root —
becomes NaN at once; the renderer drops that world. It "comes back" only at the 500-step episode
timeout, because every other termination term is a threshold comparison and comparisons against
NaN are `False`, so the broken world could never terminate early.

The trace shows sand ejected out of the bed beforehand: one world had a cluster stuck at z = 0.65 m
moving at a constant 4.2 m/s for ~50 steps before the state went NaN, with particle *velocities*
still finite. This looks like particles trapped inside the inflated foot collider.
`project_outside_colliders`, which would push them back out, is explicitly rejected inside a
coupled entry (`NotImplementedError`, `isaaclab_contrib/coupling/coupler.py:158`), so there is no
recovery path for a trapped particle.

**Ruled out.**
- Not a sparse-grid capacity overrun — `check_sparse_grid_rebuild_status()` stayed clean.
- Not curable by clamping particle speed: `particle_max_velocity` 3.0 survived 300 steps but blew
  up at step 167 over 600 steps; 5.0 blew up at 135; the default 20 at 58–129. It only shifts timing.
- Extra MPM substeps did not help (blow-up at step 129).

**Trigger identified and resolved.** 600-step runs at 4 envs:

| foot margin | sinkage (foot_z @ step 20, nominal 0.048) | reaction per foot | first NaN |
| --- | --- | --- | --- |
| 0 | −0.205 (bed bottom) | 0–80 N | none in 600 steps |
| 0.0375 | −0.048 | 300–340 N | none in 600 steps |
| 0.05 | +0.001 | 150–350 N | every 60–170 steps |

The shipped configuration was then re-checked end to end without any bench overrides — 4 envs,
900 steps, `first_nan_step=none` — which also exercises the new termination and reward scrubbing.

`FOOT_CONTACT_MARGIN` is therefore `0.75 * MPM_VOXEL_SIZE`: it keeps the bearing capacity that
2b bought (~5 cm sinkage instead of bottoming out) while staying clear of the instability. The
fraction is a stability limit, not a modelling choice, and is documented as such in `scene_cfg.py`.

**Changes made (containment, not cure).**

| File | Change |
| --- | --- |
| `mdp/terminations.py` | New `root_state_not_finite` term: `~torch.isfinite(root_state_w).all(dim=-1)`. |
| `mdp/__init__.pyi` | Exported `root_state_not_finite`. |
| `env_cfg/termination_cfg.py` | Added `solver_diverged` termination so a blown-up world resets on the same step. |
| `g1_mpm_env.py` | `step()` override scrubbing NaN/±inf from the reward buffer. Isaac Lab resets terminated envs *before* computing the returned observation, so observations are already clean, but the reward is computed pre-reset and would be NaN — one NaN sample destroys a policy update. |

---

## 4. Documentation

- `README.md` tuning notes now record the three hard constraints: `FOOT_CONTACT_MARGIN` = 0.75
  voxel (enough occupancy to carry load, below the value that traps particles), `grid_padding`
  must stay 0 on a sparse grid, and `critical_fraction` must stay 0.0 for the current sampling.
- `changelog.d/g1-29dof-mpm-locomotion.rst` describes the task as a whole (the package is still
  untracked, so no separate fragment was added for this tuning).

---

## Open items

1. **NaN blow-up resolved at margin 0.0375**, with `solver_diverged` + reward scrubbing as a
   backstop if one still occurs. If more bearing capacity is ever needed, `transfer_scheme="apic"
   → "pic"` (what `example_mpm_anymal.py` uses) is the untested lever that might allow a larger
   margin.
2. `uv run isaaclab -f` has not been run — `ruff` is not installed in either the uv venv or the
   container venv. Edited files were checked to be within 120 columns by hand.
3. Temporary scripts `scripts/_mpm_bench_tmp.py` and `scripts/_mpm_dump_tmp.py` were deleted;
   `scripts/_mpm_diag_tmp.py` (the NaN tracer) is still present and **must be deleted before
   commit**.
4. End-to-end validation of the shipped config is **done**: 4 envs, 900 steps, no bench overrides,
   `RESULT tag=shipped first_nan_step=none steps=900`. (An earlier attempt at the same run exited
   immediately with code 1 and no output — a container hiccup, not a config problem; the re-run
   above is the valid one.)
