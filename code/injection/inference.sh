#!/bin/sh
  
#SBATCH --job-name=test
#SBATCH -p iaifi_gpu 
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 01-00:00

export JAX_ENABLE_X64=True

python -u inference.py \
    --outdir ../../data/inference/injection/test \
    --posteriors ../../data/pe/tests/260207/posteriors.hdf5 \
    --injections ../../data/vt/tests/260213/detectable.hdf5 \
    --truths ./parameters/astro-o4a-strong-maxl-sigma-spin-1e-2-xi-0p3.json \
    --seed 42
