#!/bin/sh
  
#SBATCH --job-name=test
#SBATCH -p iaifi_gpu 
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 00-03:00

export JAX_ENABLE_X64=True

seed=42
outdir="../../data/inference/injection/tests"
outdir="${outdir}/260207/seed${seed}"

python -u inference.py \
    --outdir $outdir \
    --posteriors ../../data/pe/tests/260207/posteriors.hdf5 \
    --injections ../../data/vt/tests/260213/detectable.hdf5 \
    --truths ./parameters/astro-o4a-strong-maxl-sigma-spin-1e-2-xi-0p3.json \
    --seed $seed \
    > $outdir/log.out 2> $outdir/log.err
