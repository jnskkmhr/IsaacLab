* want to improve symmetry augmentation code to better handle generic observation 
* Right now, symmetry augmentation function handles hard-coded observations like source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/g1_29dof_rigid/mdp/symmetry.py
* thinking about adding MirrorObservationTerm and MirrorActionTerm so that symmetry augmentation code is more modular
    * Each observation and action term will get additional field called mirror: callable
    * Depending on the observation/action types, but generic mirror functions would be like vec3d, quat, joint
    * We can also accept function that does not do mirror if we cannot define mirror for observation terms
* Can we implement this augmentation class in source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/vel_mdp directory? 