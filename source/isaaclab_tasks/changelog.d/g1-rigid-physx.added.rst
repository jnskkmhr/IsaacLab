* Added standalone G1 29-DoF flat and rough PhysX locomotion tasks porting the
  ``ral26_gm_loco`` rigid PPO recipe, using the requested URDF-derived USDA asset.
* Added a contact-sensor-based no-flight penalty to these rigid PhysX tasks,
  matching the soft-contact tasks below 1 m/s commanded horizontal speed.
