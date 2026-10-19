# Consolidation record — 2026-10-19

Canonical source: `IsaacLab-particle-nerd/examples/particle_nerd` on branch
`junnosuke/particle_nerd`. No framework source was changed.

| Former source | Current source |
| --- | --- |
| External `experiments/particle_gnn/preview.py`, `collect_campaign.py` | `data_collection/collect.py`, `campaign.py` |
| Flat `env_cfg.py`, `scene_cfg.py`, `newton_access.py`, `terms/` | `data_collection/` |
| External GNN model, feature, training, inference, selector and evaluation modules | `gnn/` |
| Force kernels embedded in the old `gpu_kernels.py` | `force_estimation/kernels.py`, exposed through `estimator.py` |
| Flat analytical reconstruction and geometry helpers | `force_estimation/` (offline reference) |
| Flat `newton_rigid.py` | `runner/rigid.py` |
| External OSMO source staging and workflow templates | `osmo/` |

The old coupled demo's GNN checkpoint format differs from the current 50 Hz model.
Its code, old studies, benchmarks and tests for the retired model were preserved
in the archive rather than mixed into the active package. The current-model
`runner/simulation.py` connection is new; it must not be described as a validated
continuation of the old demo's accuracy results.

The backup contains both full pre-migration source trees, including uncommitted
changes, excluding caches and already-generated cloud bundles:

`/home/jkamohara/isaac/particle_nerd/archive/before-unify-20261019-151923/`

The collector's single-preview default box mass changed from 1.5 to 7.5 kg to match
the approved 5–10 kg range. Campaign sampling, dataset format, model weights and
training settings were preserved. All checks below use temporary output under
`/tmp`; original checkpoints and scientific results are untouched.

## Verification

- 25 physics/interface tests passed: contact impulse and Coulomb friction,
  contact geometry, rigid-body force application, and coupled GPU free flight.
  The GPU integration test rejects `.cpu()`, `.numpy()` and `.item()` calls
  during warmed simulation steps.
- Existing GNN radius-neighborhood, distant-body influence, graph padding and
  particle-history checks passed. Activity-selection checks passed, including
  exact preservation of inactive positions and transient-motion labels.
- The original and moved motion networks produced bitwise-identical outputs
  with the existing checkpoint, identical graph inputs and deterministic Torch
  aggregation. Separate three-step selected rollouts were not bitwise identical:
  maximum coordinate/velocity differences were 0.131 mm and 0.00437 m/s. Repeating
  the original code alone gave 0.088 mm and 0.00430 m/s. Thus the full default GPU
  rollout is not deterministic; exact end-to-end equality is not claimed.
- A real MPM/MJWarp collection completed for a 7.5 kg box, including settling and
  a 20 ms post-settle recording. A one-step small-network training smoke test
  completed validation and checkpoint writes using two distinct temporary data
  entries. This is execution verification, not a new training result.
- The actual current GNN and activity checkpoints completed five headless coupled
  steps with finite states, no graph overflow, and converged contact solves.
  This is not a long-horizon accuracy test. Interactive mouse input was wired to
  the viewer but not manually exercised during this consolidation.
- Current-format offline force evaluation ran successfully. Its short smoke run
  is not a force-accuracy benchmark.
- OSMO staging built a roughly 180 MB archive containing one canonical example
  and the tested backend. Collection and training templates parse after their
  Jinja/runtime substitutions. No cloud job was launched.
- Example-specific formatting, lint, YAML, license and whitespace checks pass.
  The repository-wide changelog-fragment hook fails on pre-existing invalid or
  missing fragments in `source/isaaclab*`. Those framework files were not changed.

New recordings also include `body_contact_friction=0.6`, taken from the existing
collector's rigid-body configuration. This is additive metadata; the current
runner and force evaluator use 0.6 for older recordings without that field.
