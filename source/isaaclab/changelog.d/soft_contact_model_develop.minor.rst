Added
^^^^^

* Added :attr:`~isaaclab.terrains.TerrainImporterCfg.disable_collider`, which imports the terrain mesh as a
  visual-only prim. This is useful when the ground reaction is supplied by a separate contact model instead
  of the physics engine.
* Added :attr:`~isaaclab.terrains.trimesh.MeshPlaneTerrainCfg.ground_height_range`, which places the plane
  mesh at a difficulty-interpolated height. It defaults to ``(0.0, 0.0)``, which keeps the previous behavior
  of a plane at ``z = 0``.
