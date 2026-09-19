Added
^^^^^

* Added the NeRD neural dynamics package to the develop uv workspace, including
  dataset generation, dynamics training, evaluation, and neural RSL-RL tasks.

Changed
^^^^^^^

* Migrated task imports, launcher helpers, packaging, and Newton manager hooks
  to develop. Use ``uv sync``, ``--visualizer none``, and ``newton_mjwarp`` for
  physical-solver presets; omit physics presets for concrete NeRD tasks.

Fixed
^^^^^

* Invalidated Newton forward kinematics after direct generalized-state writes
  so dataset and neural environment resets refreshed root-body transforms.
* Preserved post-step callbacks and NeRD history reset ordering with the
  current Newton manager lifecycle.
