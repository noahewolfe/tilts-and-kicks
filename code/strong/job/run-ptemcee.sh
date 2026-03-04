#!/bin/bash

export JAX_ENABLE_X64=1
unset PYTHONPATH

outdir="../../data/inference/strong/multimass/noah/twomass/ptemcee"

mkdir -p $outdir

/work/submit/newolfe/miniforge3/envs/just-for-kicks-260228/bin/python -u ptemcee-multimass-gwpop.py \
    --outdir $outdir \
    --which-data 'noah' \
    --model 'twomass' \
    --priors './priors/twomass-only-mass.prior' \
    --sampling-seed 746566 \
    --maximum-uncertainty 1 \
    --stable-expit \
    > $outdir/log.out 2> $outdir/log.err
