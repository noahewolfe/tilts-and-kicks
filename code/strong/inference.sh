#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export JAX_ENABLE_X64=1

outdir='../../data/inference/strong/onemass-xphm-gwtc3-var1'
mkdir -p $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u inference.py \
    --outdir $outdir \
    --priors ./priors/stegmann.prior \
    --nprior 0 \
    --nlive 150 \
    --maximum-variance 1 \
    #> $outdir/log.out 2> $outdir/log.err
