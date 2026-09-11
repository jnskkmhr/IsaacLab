<!-- ## What do I want to do ?
* Implement g1_29_dof locomotion environment with Mjwarp + MPM solver soupled with Proxy/ADMM solver coupler.
* Locomotion policy formulation is from /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/g1_29dof_soft
* MPM + Mjwarp coupling code can be referred to /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/ur10_particle_push or /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour
* Also, I have standalone newton implementation, so you can refer how I spawn MPM sand (coupling should be done by ProxyCoupling or ADMM coupling, see how coupling is done in two environments mentioned above). /home/jkamohara/isaac/newton/newton/examples/mpm/g1

## How to implement?

### RL formulation
* action, command, curriculum, termination can be cloned from g1_29dof_soft as these terms may not be dependent on physics solver
* event terms depends on physics solver, but I would drop terrain parameter randomization for now to simplify code
* observation is tricky. Privileged information includes terrain information. You can think about how to retrieve these from mpm solver
* reward is also tricky. You can ignore reward functions that requires contact state.
* scene config should be implemented by referring to franka_pur/ur10_particle_push

* When you access to MPM sand state that are used for reward/observation/event, it might be better to add additional method to environment class that override manager_based_rl_env.py. This is because each method in env class can bookeep MPM variables. If you implement each terms as separate class, you cannot bookeep, thus you end up computing the same variables multiple times (contact force computed from MPM contact impulse for example). I think good example is `_particle_position_e` in ur10_particle_push_env.py

### Physics
* look at how physics are coupled in franka_pur/ur10_particle_push
```python
        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name=RIGID_ENTRY,
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False,
                            integrator="implicitfast",
                            njmax=256,
                            nconmax=512,
                        ),
                        bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Table"],
                        include_static_shapes=True,
                        # Refine rigid integration with three substeps per coupled interval.
                        substeps=3,
                    ),
                    CouplerEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            voxel_size=MPM_VOXEL_SIZE,
                            grid_type="sparse",
                            grid_padding=0,
                            strain_basis="P0",
                            transfer_scheme="apic",
                            # Preserve particle-backed constitutive history on the rebuildable sparse grid.
                            max_iterations=24,
                            tolerance=1.0e-4,
                            warmstart_mode="auto",
                            velocity_basis="Q1",
                            collider_basis="pic27",
                            collider_velocity_mode="forward",
                            solver="auto",
                            separate_worlds=True,
                            project_outside_colliders=False,
                        ),
                        bodies=[
                            r"/World/envs/env_.*/MPMWorkSurface",
                            r"/World/envs/env_.*/MPMGround",
                            r"/World/envs/env_.*/MPMBinFloor",
                            r"/World/envs/env_.*/MPMBinFront",
                            r"/World/envs/env_.*/MPMBinBack",
                            r"/World/envs/env_.*/MPMBinLeft",
                            r"/World/envs/env_.*/MPMBinRight",
                        ],
                        all_particles=True,
                        include_static_shapes=False,
                        include_child_joints=False,
                        # Keep entry-local substeps at one; collider poses refresh between outer coupled substeps.
                        substeps=1,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source=RIGID_ENTRY,
                        destination=MPM_ENTRY,
                        bodies=[r"/World/envs/env_.*/Robot/ee_link/Paddle"],
                        mode="lagged",
                        mass_scale=self.proxy_mass_scale,
                        collision_pipeline=None,
                    )
                ],
                iterations=1,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            # Run one coupled solve per 120 Hz simulation step.
            num_substeps=1,
            # Bounded sparse topology defers capture until the initial reset has authored its graph shape.
            use_cuda_graph=True,
        )
        # Keep the UR10's implicit drives on the Newton backend used by the coupled step.
        self.sim.use_newton_actuators = True
        configure_sparse_mpm_capacities(self)
``` -->


* Correct privileged observation misalignment with g1_29dof_soft.
* privileged observation in g1_29dof_soft is
```python
@configclass
class PrivilegedObsCfg(ObsGroup):
    """Observations for policy group."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    foot_height = ObsTerm(
        func=g1_mdp.foot_height,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )

    foot_contact = ObsTerm(
        func=g1_soft_mdp.foot_contact_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_threshold": 5.0,
            "soft_force_threshold": SOFT_CONTACT_THRESHOLD,
        },
    )
    foot_contact_force = ObsTerm(
        func=g1_soft_mdp.foot_contact_forces_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_filter_threshold": 5.0,
            "soft_force_filter_threshold": SOFT_CONTACT_THRESHOLD,
        },
    )
    foot_air_time = ObsTerm(
        func=g1_soft_mdp.foot_air_time_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

    terrain_material_parameters = ObsTerm(
        # func=mdp.terrain_material_parameters_all_hybrid,
        func=g1_soft_mdp.terrain_material_parameters_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

```
and g1_29dof_mpm has the following privileged information:
```python
@configclass
class PrivilegedObsCfg(ObsGroup):
    """Granular-terrain information available to the critic only."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    foot_height = ObsTerm(
        func=g1_mdp.foot_height,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    foot_sinkage = ObsTerm(func=mpm_mdp.foot_sinkage)
    foot_contact = ObsTerm(func=mpm_mdp.foot_contact)
    foot_air_time = ObsTerm(func=mpm_mdp.foot_air_time)
    # local terrain shape, the granular replacement for the ray-caster height scan
    sand_height_scan = ObsTerm(func=mpm_mdp.sand_height_scan, clip=(-1.0, 1.0))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1
```

`foot_sinkage` and `sand_height_scan` came out of nowhere. Please correct them.
Also, current observation is missing foot_contact_force and terrain_material_parameters. If you do not know how to deal with terrain material parameters, just implement mdp function that returns zero tensor with the same shape.
