#!/bin/bash
#SBATCH -p iaifi_gpu_requeue  # partition
#SBATCH --mem=25GB        # memory in GB
#SBATCH --time=00:00:20 # time in HH:MM:SS
#SBATCH -c 5          # number of cores

export JAX_ENABLE_X64=True

#outdir='../../data/inference/injection/tests/260207/nobs70-seed746569-ulin-broad'
outdir='/n/home03/newolfe/projects/tilts-and-kicks/data/inference/injection/salvo-iso/nobs150-seed1768-ulin-broad'

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u chieff_ppd.py $outdir #> $outdir/mc.out 2> $outdir/mc.err
