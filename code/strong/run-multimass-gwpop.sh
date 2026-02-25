#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export JAX_ENABLE_X64=1

data="stegmann"
#model="default-spin-simple-power-law-mass"
#model="default-spin-bpl2p-mass"
#model="twomass"
model="threemass"
var=1

outdir="../../data/inference/strong/multimass/${data}/${model}-var-${var}-stableexpit"
mkdir -p $outdir

echo $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u multimass-gwpop.py \
    --outdir $outdir \
    --which-data $data \
    --model $model \
    --maximum-uncertainty $var \
    --priors ./priors/threemass-only-mass.prior \
    > $outdir/log.out 2> $outdir/log.err
