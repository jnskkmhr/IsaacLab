# PACE setup 

### Write ssh-config 

```bash 
# run vi ~/.ssh/config
Host pace-phoenix
        HostName login-phoenix.pace.gatech.edu
        User your-gt-username # for example jkamohara3

Host pace-ice
        HostName login-ice.pace.gatech.edu
        User your-gt-username
```

### Install docker and nvidia docker runtime

Run following to install docker if you do not have one. 

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# install docker
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Follow the link to install nvidia docker runtime. 

[Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian)

[Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)

## Build singularity image

### build docker image first

```bash
./docker/container.py start
```

### convert docker image to singularity sif image
Apptainer steup 

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

Convert docker image to sif file and upload it to  `CLUSTER_SIF_PATH`  in `docker/cluster/env.cluster`

```bash
./docker/cluster/cluster_interface.sh push 
```

## Edit cluster settings (VIP student: start from here)

Edit the following files to match your PACE crendentials 

<!-- - `docker/cluster/submit_job_slurm.sh` -->
- `docker/cluster/.env.cluster`
- `docker/cluster/.env.etc`

For example, you can change the following environment variables to match your pace account. 

```
# in docker/cluster/.env.cluster
CLUSTER_ROOT_DIR=/storage/ice1/6/4/jkamohara3/Research
CLUSTER_ISAAC_SIM_CACHE_DIR=$CLUSTER_ROOT_DIR/g1_loco_rigid/docker-isaac-sim
CLUSTER_ISAACLAB_DIR=$CLUSTER_ROOT_DIR/g1_loco_rigid/isaaclab
```

Make sure your directory exists. 
For `CLUSTER_ROOT_DIR`, you can run `pace-quota` in PACE and find your storage directory. Here is my example. 

```bash
Filesystem                                             Usage (GB)    Limit
Home:/home/hice1/jkamohara3                                   0.0     30.0   0.0%   
Scratch:/storage/ice1/6/4/jkamohara3                         18.0    300.0   6.0%
```

Then, make necessary directories. 
```bash
cd /your/scratch/directory 
mkdir Research 
cd Research 
# example
mkdir g1_loco_rigid
mkdir g1_loco_soft
```

Lastly, replace wandb api key in `.env.etc` to yours. \
We are all set to run interactive job!


## Run interactive job (functional, but rendering not working)

First push your current code to pace 
```bash 
./docker/cluster/cluster_interface.sh sync
```

Then ssh to pace-ice and start byobu session (terminal manager). 
```bash 
byobu new -s pace-training 
ssh pace-ice 

# then move to your isaaclab directory in PACE. 
cd /your/scratch/directory
cd Research/g1_loco_rigid/isaaclab_{timestamp}

# for pace-ice 
./docker/cluster/start_interactive_node_ice.sh
# for pace-phoenix
./docker/cluster/start_interactive_node_phoenix.sh
```

After interactive job is initialized, run the training command 
```bash
# vanilla PPO on rigid terrain
./docker/cluster/run_singularity.sh $PWD isaac-lab-base --task Isaac-Velocity-Flat-G1-29dof-v1 --num_envs 4096 --headless
# vanilla PPO on soft terrain
./docker/cluster/run_singularity.sh $PWD isaac-lab-base --task Isaac-Velocity-Flat-G1-29dof-Soft-v1 --num_envs 4096 --headless --video --enable_cameras

# teacher policy on soft terrain
./docker/cluster/run_singularity.sh $PWD isaac-lab-base --task Isaac-Velocity-Flat-G1-29dof-Soft-Teacher-v1 --num_envs 4096 --headless --agent rsl_rl_distillation_cfg_entry_point
# student BC distillation on soft terrain
./docker/cluster/run_singularity.sh $PWD isaac-lab-base --task Isaac-Velocity-Flat-G1-29dof-Soft-Student-v1 --num_envs 4096 --headless --agent rsl_rl_distillation_cfg_entry_point \
--checkpoint /workspace/isaaclab/logs/rsl_rl/g1_29dof_soft_vae_teacher/2026-05-02_11-26-49/model_24000.pt
# student BC distillation + PPO on soft terrain
./docker/cluster/run_singularity.sh $PWD isaac-lab-base --task Isaac-Velocity-Flat-G1-29dof-Soft-Student-v1 --num_envs 4096 --headless --agent rsl_rl_ppo_distillation_cfg_entry_point \
--checkpoint /workspace/isaaclab/logs/rsl_rl/g1_29dof_soft_vae_teacher/2026-05-02_11-26-49/model_24000.pt

# WBC training
./docker/cluster/run_singularity.sh $PWD isaac-lab-base --task Motion-Tracking-G1-v0 --num_envs 4096 --headless
```


## Run batch job (not working....)

```bash 
# G1 whole body tracking 
./docker/cluster/cluster_interface.sh job --task Motion-Tracking-G1-v0 --num_envs 4096 --headless --video --enable_cameras

# G1 soft terrain locomotion 
./docker/cluster/cluster_interface.sh job --task Isaac-Velocity-Flat-G1-29dof-Soft-v0 --num_envs 4096 --headless --video --enable_cameras
```