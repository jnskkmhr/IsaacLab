#!/usr/bin/env bash

# Phoenix
# cat <<EOT > job.sh
# #!/bin/bash

# #SBATCH --account=gts-yzhao301
# #SBATCH --cpus-per-task=12
# #SBATCH --gpus-per-node=rtx_pro_6000_blackwell:1
# #SBATCH --mem-per-gpu=24G
# #SBATCH --time=2:00:00
# #SBATCH --nodes=1
# #SBATCH -qinferno
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=jkamohara3@gatech.edu
# #SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# # Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
# bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
# EOT

# sbatch < job.sh
# rm job.sh

# ICE
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=rtx_pro_6000_blackwell:1
#SBATCH --mem-per-gpu=24G
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -q=coe-grade
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jkamohara3@gatech.edu
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh