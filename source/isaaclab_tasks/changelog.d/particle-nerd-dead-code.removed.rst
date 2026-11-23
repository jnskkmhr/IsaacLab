* Removed unused options of the Particle NeRD model and solver: the ballistic update rule, the stress contact
  head, the model's own active-particle selection, the signed-distance-grid collider input, the bed settling
  step, and the solver configuration fields ``settle_steps``, ``settle_cache_dir``, ``batch_inference``,
  ``use_unified_sdf_colliders``, ``use_stress_contact_head``, ``gradient_radius``, ``far_support_scale``,
  ``far_gap_weight`` and ``far_influence_weight``. The far-field settings are read from the checkpoint.
