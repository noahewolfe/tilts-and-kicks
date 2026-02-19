#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export JAX_ENABLE_X64=1

outdir='../../data/inference/strong/spl'
mkdir -p $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u spl.py \
    --outdir $outdir \
    --nlive 150 \
    --maximum-variance 1 \
    > $outdir/log.out 2> $outdir/log.err
