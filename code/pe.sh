#!/bin/bash
#SBATCH --job-name=pe
#SBATCH --partition=mit_normal
#SBATCH --time=00-12:00
#SBATCH --mem=4GB
#SBATCH -c 1 
#SBATCH --array=0-99
#SBATCH --requeue

outdir="../data/pe/snr15/ct-0p9"
outdir="${outdir}/${SLURM_ARRAY_TASK_ID}"

echo $outdir
mkdir -p $outdir

#py="/work/submit/newolfe/miniforge3/envs/just-for-kicks/bin/python"
py="/home/newolfe/.conda/envs/just-for-kicks/bin/python"
timeout 8h $py pe.py \
    --outdir $outdir \
    --npool 1 \
    --prior-path ./priors/bbh.prior \
    --event-index $SLURM_ARRAY_TASK_ID \
    --time-reference H1 \
    --nlive 1000 \
    --catalog-path ../data/vt/snr15/ct-0p9/detectable.hdf5 \
    >> "${outdir}/log.out" 2>> "${outdir}/log.err"

if [[ $? == 124 ]]; then 
    scontrol requeue $SLURM_JOB_ID
fi
