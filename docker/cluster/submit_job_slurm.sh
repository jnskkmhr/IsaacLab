#!/usr/bin/env bash
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --account=gts-yzhao301
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time=48:00:00
#SBATCH -N1
#SBATCH -qinferno
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jkamohara3@gatech.edu
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT
cat < job.sh
sbatch < job.sh
rm job.sh