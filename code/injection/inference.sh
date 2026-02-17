#!/bin/sh
  
#SBATCH --job-name=iso
#SBATCH -p iaifi_gpu_requeue 
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 00-03:00

set -euox pipefail

export JAX_ENABLE_X64=True

nobs=150
seed=1764
outdir="../../data/inference/injection/tests"
outdir="${outdir}/salvo-iso/nobs${nobs}-seed${seed}-ulin-broad"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u inference.py \
    --outdir $outdir \
    --posteriors ../../data/pe/salvo-iso/posteriors_from_ids1.pkl \
    --injections ../../data/vt/zenodo.17080422/injections.h5 \
    --truths ./parameters/astro-o4a-strong-maxl-sigma-spin-1e-2-xi-0p3.json \
    --seed $seed \
    --maximum-variance 5 \
    --nobs $nobs \
    > $outdir/log.out 2> $outdir/log.err
