#!/bin/bash
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia_a30
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-0               # update to match CONFIGS length below
#SBATCH --output=../../data/inference/strong/identifiable/slurm-%A_%a.out
#SBATCH --error=../../data/inference/strong/identifiable/slurm-%A_%a.err

set -euo pipefail

CONFIGS=(
    configs/identifiable.sh
    # add entries here, increment --array upper bound above
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
source "$CONFIG"
mkdir -p "$outdir"

SCRIPT_DIR="/home/submit/newolfe/projects/tilts-and-kicks/code/strong"
SIF="/home/submit/newolfe/projects/tilts-and-kicks/containers/almalinux9.sif"

echo "Array task ${SLURM_ARRAY_TASK_ID}: config=${CONFIG}, outdir=${outdir}"

apptainer exec \
    --nv \
    --bind /home/submit/newolfe \
    --bind /work/submit/newolfe \
    --bind /usr/bin/git:/usr/bin/git \
    --pwd "${SCRIPT_DIR}" \
    "${SIF}" \
    bash ${SCRIPT_DIR}/job/run-identifiable-gwpop.sh "$CONFIG"
