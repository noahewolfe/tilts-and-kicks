#!/bin/bash
#SBATCH --partition=submit-gpu-express
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia_a30
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=12:00:00
# --job-name, --output, --error, --array set by launch-array.sh

set -euo pipefail

IFS=':' read -ra _vals <<< "${SWEEP_VALUES:?Must set SWEEP_VALUES}"
_val="${_vals[$SLURM_ARRAY_TASK_ID]}"
_base_config=$(realpath "${BASE_CONFIG:?Must set BASE_CONFIG}")

source "$_base_config"
task_outdir="${outdir}/${SWEEP_PARAM}/${_val}"
mkdir -p "$task_outdir"

# Redirect all output to per-task log files
exec 1>"$task_outdir/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
exec 2>"$task_outdir/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# Write self-documenting per-task config
_task_config="$task_outdir/config.sh"
{
    printf '# Auto-generated for array task %s\n' "$SLURM_ARRAY_TASK_ID"
    printf 'source %s\n' "$_base_config"
    printf '%s=%s\n' "$SWEEP_PARAM" "$_val"
    printf 'outdir=%s\n' "$task_outdir"
} > "$_task_config"

echo "=== Array task ${SLURM_ARRAY_TASK_ID} ==="
echo "Job      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node     : $(hostname)"
echo "Sweep    : ${SWEEP_PARAM}=${_val}"
echo "Outdir   : ${task_outdir}"
echo "Started  : $(date)"

SCRIPT_DIR="/home/submit/newolfe/projects/tilts-and-kicks/code/strong"
SIF="/home/submit/newolfe/projects/tilts-and-kicks/containers/almalinux9.sif"

apptainer exec \
    --nv \
    --bind /home/submit/newolfe \
    --bind /work/submit/newolfe \
    --bind /usr/bin/git:/usr/bin/git \
    --pwd "${SCRIPT_DIR}" \
    "${SIF}" \
    bash job/run-multimass-gwpop.sh "$_task_config"

echo "Finished : $(date)"
