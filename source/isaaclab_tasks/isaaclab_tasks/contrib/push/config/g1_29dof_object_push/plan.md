Box pushing 
* Now we have basic policy for locomotion (source/isaaclab_tasks/isaaclab_tasks/contrib/push/config/g1_29dof_locomotion) and manipulation (source/isaaclab_tasks/isaaclab_tasks/contrib/push/config/g1_29dof_manipulation)
* Next thing we can do is distilling these two policies into loco-manipulation policy where command input is box velocity. Motivation is I want to remove global pose estimation of robot/box

Student observes 
* Robot's global pose / odometry
* Box's pose wrt floating base 
* Proprioception 

Student's reward 
* box velocity tracking like source/isaaclab_tasks/isaaclab_tasks/contrib/push/config/g1_29dof_mpm_object_push_old

Teacher 
* Multiple policies (locomotion and manipulation)
* During distillation, we need to generate locomotion's velocity command and manipulator's hand SE3 pose
    * velocity command could be same as box velocity command 
    * hand pose -> can pick two fixed points on box face that is closest to robot

Overall implementation 
* need to add ability to load multiple teacher policies in rsl-rl or compositional-rsl-rl, might be better to make this change in compositional-rsl-rl as network construction/policy loading can be taken care by comp-rsl-rl
* it will be also nice if we can add PPO on top of distillation. Well, if we treat distillation as supervised learning on teacher, can we support distillation as auxililary task on top of normal PPO.
* make all changes in https://github.com/jnskkmhr/compositional_rsl_rl/tree/main or is it better to do in flash-sac branch, idk. 