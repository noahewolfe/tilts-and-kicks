#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export JAX_ENABLE_X64=1

outdir='../../data/inference/strong/twomass'
mkdir -p $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u inference.py \
    --outdir $outdir \
    > $outdir/log.out 2> log.err