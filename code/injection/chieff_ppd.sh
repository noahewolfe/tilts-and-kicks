#!/bin/bash
#SBATCH -p serial_requeue  # partition
#SBATCH --mem=25GB        # memory in GB
#SBATCH --time=00:01:00 # time in HH:MM:SS
#SBATCH -c 1           # number of cores

export JAX_ENABLE_X64=True

outdir='../../data/inference/injection/tests/260207/nobs70-seed746569-ulin-broad'

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u chieff_ppd.py $outdir #> $outdir/mc.out 2> $outdir/mc.err
