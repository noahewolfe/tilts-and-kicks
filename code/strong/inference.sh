#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export JAX_ENABLE_X64=1

outdir='../../data/inference/strong/threemass'
mkdir -p $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u inference.py \
    --outdir $outdir \
    --priors ./priors/threemass.prior \
    --nprior 100_000 \
    > $outdir/log.out 2> $outdir/log.err
