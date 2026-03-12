#!/bin/bash
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia_a30
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=2-00:00:00
# --job-name, --output, --error set by caller

set -euo pipefail

CONFIG=${CONFIG:?Must set CONFIG env var before sbatch}

SCRIPT_DIR="/home/submit/newolfe/projects/tilts-and-kicks/code/strong"
SIF="/home/submit/newolfe/projects/tilts-and-kicks/containers/almalinux9.sif"

echo "=== Job info ==="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Config   : ${CONFIG}"
echo "Started  : $(date)"

apptainer exec \
    --nv \
    --bind /home/submit/newolfe \
    --bind /work/submit/newolfe \
    --bind /usr/bin/git:/usr/bin/git \
    --pwd "${SCRIPT_DIR}" \
    "${SIF}" \
    bash ${SCRIPT_DIR}/job/run-identifiable-gwpop.sh "$CONFIG"

echo "Finished : $(date)"
