Let's implement general whole body control that can track arbitrary task space command.
Motion would not be super crazy ones like backflip and dance.
I would formulate the tracking task as minimize tracking error of
* SE3 pose of floating base (either torso or pelvis)
* SE3 pose of each foot (ankle roll link)
* SE3 pose of each end effector (wrist yaw link)
And I want to achieve diverse motion with this policy like walking, squat, squat with waist twisting etc.

Training pipeline is teacher student distillation or twin-model (will explain detail below).

## Teacher-Student policy
### Teacher
* We pre-generate a bunch of whole body pose data that is collisison free and singularity free. This can be mass-produced by Mink IK/or NVIDIA Soma. Then, we have so called whole body pose dataset that consists of joint pos, joint vel, and body (link) pose.
    * I would say you fix left foot and randomize the rest of dofs to generate kinematically feasible whole body pose dataset
    * save this data as npz like mimic task
    * implement dataset loader class that process single step target pose (most of code can be borrowed from  mimic code.)
* Actor is trained with proprioception (IMU like projected gravity, measured joint pos and velocity), previous action, and reference joint position (target whole body jont pos).
* Critic can take additional privileged information like joint vel and target  body pose. Basically, you can look at how I do in mimic task.

### Student
* Student is trained with a distillation algorithm implemented in rsl-rl
* We command policy via task space command (not whole body pose) like SE3 pose of floating base, foot, and hand (end effector)
* While distillation, you can random sample motion dataset and pick this task space pose. This task space is sent to student policy while full whole body pose are sent teacher policy.
* Student observation is proprioception, last action, and task space command.

## Twin model
* Instead of doing two-stage distillation, I would have model consists of teacher encoder, student encoder, and shared action decoder. Teacher encoder takes teacher observation and output h-dimensional latent  embedding z_teacher. Student encoder takes student observation and output same dimension  latent embedding z_student. Then, you feed this embedding to shared action decoder and get action a_teacher (=pi(z_teacher)) and action a_student (=pi(z_student)). Note that pi is shared.
* Difficulty is which action we would use for rollout (to estiamte advantage)
* When sharing action decoder, how do we backpropagate ? Like when you forward pi 2 times from z_teacher and z_student, can we do packprop?
* Then, adding teacher-student representation matching loss makes sense ? L=|a_student - a_teacher| ?
* But, this might fundamentally solve teacher-student  two stage training complexity and hope teacher-student observation mismatch.


## Other details
### Dataset  generation
* I would generate at least 100K different whole body pose data. Then, you can mirror left/right to create exact symmetry version of original data
### Symmetry
* Use symmetry augmentation for training
### Code structure
* Make wbc task standalone, do not inherit any class from other task like (mimic and velocity) and do not re-use mdp functions. Just define utility mdps in source/isaaclab_tasks/isaaclab_tasks/contrib/wbc/mdp and task for g1 in source/isaaclab_tasks/isaaclab_tasks/contrib/wbc/config/g1_29dof
* naming of files/variable very clear. Avoid uncommon terms, and do not abbreviate owner of word like (center -> idk what's object center. Say robot_center, object_center etc. This is just example)

## Testing order
1. Use mink/soma to generate feasible whole body motion data. for robot, you can use /home/jkamohara/isaac/newton-assets/unitree_g1/urdf/g1_29dof_rev_1_0_box_foot_improved_collision.urdf as asset. just generate small data for sanity check
2. Implement teacher-student method first and sanity check training with data from 1
3. Mass produce whole body dataset
4. Train teacher policy
5. Then, distill teacher to student.

I would say you can do 4 during overnight. You may need my input before you begin 5 as student depends on teacher policy quality.

You can leave me comments for twin model structure. What;s your suggestion and thoughts for my concerns and something else that I am missing.
