* Idk isaaclab repo has GO2 asset, if not find from https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go2_description and convert to usda
* follow env construction rule written in skill (defined in T2)
* write minimal mdp like reward, observation, event, curriculum etc to make it walk. Does unitree lab has any GO2 policy recipe ?
* use both fixed mpm and streaming mpm to construct environment
* follow exactly same env structure as g1_29dof
* We should share utilities like /home/jkamohara/isaac/worktrees/mpm-unification/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity_mpm/config/g1_29dof/util, moving this under velocity_mpm/ should do make code sharable
