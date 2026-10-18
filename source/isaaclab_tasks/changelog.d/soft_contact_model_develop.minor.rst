Added
^^^^^

* Added the ``IsaacContrib-Velocity-Flat-G1-29dof-Rigid`` and ``IsaacContrib-Velocity-Rough-G1-29dof-Rigid``
  locomotion tasks, together with their ``-Play`` variants, under
  ``isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid``. The task ships a self-contained ``mdp``
  sub-module with the gait, foot-orientation and velocity-curriculum terms it needs.
* Added the ``isaaclab_tasks.contrib.soft_contact`` package, which models the ground reaction of granular
  terrain with resistive force theory (RFT) and applies it through the
  :class:`~isaaclab_tasks.contrib.soft_contact.PhysicsCallbackActionCfg` action term.
* Added the ``IsaacContrib-Velocity-Flat-G1-29dof-Soft`` locomotion task, together with its ``-Finetune``
  and ``-Play`` variants, under ``isaaclab_tasks.contrib.velocity.config.g1_29dof_soft``. The task walks on
  granular terrain driven by the soft contact model and reuses the rigid-terrain task's MDP terms for
  everything that is not soft-terrain specific.
