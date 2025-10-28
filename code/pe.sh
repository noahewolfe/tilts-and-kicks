#!/bin/bash
#SBATCH --job-name=pe
#SBATCH --partition=submit
#SBATCH --time=06-00:00
#SBATCH --mem=8GB
#SBATCH -c 8
#SBATCH --array=0-999

outdir="/work/submit/newolfe/tilts-and-kicks/data/pe/snr15/ct-0p9"
outdir="${outdir}/${SLURM_ARRAY_TASK_ID}"

echo $outdir
mkdir -p $outdir

py="/work/submit/newolfe/miniforge3/envs/just-for-kicks/bin/python"
timeout 130h $py pe.py \
    --outdir $outdir \
    --npool 8 \
    --prior-path ./priors/bbh.prior \
    --event-index $SLURM_ARRAY_TASK_ID \
    --time-reference H1 \
    --nlive 1000 \
    --catalog-path /work/submit/newolfe/tilts-and-kicks/data/vt/snr15/ct-0p9/detectable.hdf5 \
    >> "${outdir}/log.out" 2>> "${outdir}/log.err"

if [[ $? == 124 ]]; then 
    scontrol requeue $SLURM_JOB_ID
fi
