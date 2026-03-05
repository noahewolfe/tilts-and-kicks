#!/bin/bash
#SBATCH --partition=submit-gpu-express
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia_a30
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=1:00:00
#SBATCH --error=./ptemcee.err
#SBATCH --out=./ptemcee.out

SCRIPT_DIR="/home/submit/newolfe/projects/tilts-and-kicks/code/strong"
SIF="/home/submit/newolfe/projects/tilts-and-kicks/containers/almalinux9.sif"

apptainer exec \
    --nv \
    --bind /home/submit/newolfe \
    --bind /work/submit/newolfe \
    --bind /usr/bin/git:/usr/bin/git \
    --pwd "${SCRIPT_DIR}" \
    "${SIF}" \
    bash ./job/run-ptemcee.sh 
