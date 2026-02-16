#!/bin/sh
  
#SBATCH --job-name=test
#SBATCH -p iaifi_gpu_requeue 
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 00-03:00

set -euox pipefail

export JAX_ENABLE_X64=True

nobs=70
seed=746570
outdir="../../data/inference/injection/tests"
outdir="${outdir}/260207/nobs${nobs}-seed${seed}-ulin-broad"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u inference.py \
    --outdir $outdir \
    --posteriors ../../data/pe/tests/260207/posteriors.hdf5 \
    --injections ../../data/vt/tests/260213/detectable.hdf5 \
    --truths ./parameters/astro-o4a-strong-maxl-sigma-spin-1e-2-xi-0p3.json \
    --seed $seed \
    --maximum-variance 10 \
    --nobs $nobs \
    > $outdir/log.out 2> $outdir/log.err
