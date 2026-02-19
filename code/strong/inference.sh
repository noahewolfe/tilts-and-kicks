#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export JAX_ENABLE_X64=1

outdir='../../data/inference/strong/onemass-xphm-gwtc3-var5'
mkdir -p $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u inference.py \
    --outdir $outdir \
    --priors ./priors/stegmann.prior \
    --nprior 100_000 \
    --nlive 150 \
    --maximum-variance 5 \
    > $outdir/log.out 2> $outdir/log.err
