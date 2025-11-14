#!/bin/bash
#SBATCH --job-name=vt
#SBATCH --array=0-999               # Job array of 100 jobs (IDs 0–99)
#SBATCH --time=03-00:00:00             # Walltime (hh:mm:ss)
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --partition=submit

outdir="/work/submit/newolfe/tilts-and-kicks/data/vt/snr15/ct-iso-mag-unif-broad-mass"
initseed=10300
seed=$((initseed + SLURM_ARRAY_TASK_ID))
outdir="${outdir}/${seed}"
echo $outdir
mkdir -p $outdir

py="/work/submit/newolfe/miniforge3/envs/just-for-kicks/bin/python"
$py vt.py \
    --outdir $outdir \
    --ninj 100 \
    --snr-threshold 15 \
    --seed $seed \
    --model "o4a-strong-unif-tilts-unif-mag" \
    --parameters "./parameters/o4a-strong-vt.json" \
    --extra-kwargs "{\"cos_tilt_min\" : -1, \"cos_tilt_max\" : 1}" \
    > "${outdir}/log.out" 2> "${outdir}/log.err"

