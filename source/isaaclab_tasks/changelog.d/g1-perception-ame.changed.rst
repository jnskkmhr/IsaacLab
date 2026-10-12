* Switched the G1 perception locomotion tasks to compositional RSL-RL PPO with
  attention-based XYZ height-map encoding. The former depth task IDs use the
  same height-map configuration; existing LiDAR/depth checkpoints require retraining.
* Enabled symmetry augmentation for G1 perception observations, histories, actions, and elevation maps, and added swing/stance duration variance and foot-stumble penalties inspired by AME_Locomotion. Existing AME weights remain loadable and need further training to benefit from these changes.
