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
seed=1768
outdir="../../data/inference/injection"
outdir="${outdir}/salvo-iso/nobs${nobs}-seed${seed}-ulin-broad"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u inference.py \
    --outdir $outdir \
    --posteriors ../../data/pe/salvo-iso/posteriors_from_ids1.pkl \
    --injections ../../data/vt/zenodo.17080422/injections.h5 \
    --truths ../../data/pe/salvo-iso/allinjs_list_O4.dat \
    --seed $seed \
    --maximum-variance 5 \
    --nobs $nobs \
    --cut 11 \
    > $outdir/log.out 2> $outdir/log.err
