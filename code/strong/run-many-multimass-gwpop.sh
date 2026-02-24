#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export JAX_ENABLE_X64=1

data="stegmann"
var=1

###

model="default-spin-simple-power-law-mass"

outdir="../../data/inference/strong/multimass/${data}/${model}-var-${var}"
mkdir -p $outdir

echo $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u multimass-gwpop.py \
    --outdir $outdir \
    --which-data $data \
    --model $model \
    --maximum-uncertainty $var \
    > $outdir/log.out 2> $outdir/log.err

###

model="default-spin-bpl2p-mass"

outdir="../../data/inference/strong/multimass/${data}/${model}-var-${var}"
mkdir -p $outdir

echo $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u multimass-gwpop.py \
    --outdir $outdir \
    --which-data $data \
    --model $model \
    --maximum-uncertainty $var \
    --priors ./priors/onemass-only-mass.prior
    > $outdir/log.out 2> $outdir/log.err

###

model="twomass"

outdir="../../data/inference/strong/multimass/${data}/${model}-var-${var}"
mkdir -p $outdir

echo $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u multimass-gwpop.py \
    --outdir $outdir \
    --which-data $data \
    --model $model \
    --maximum-uncertainty $var \
    --priors ./priors/twomass-only-mass.prior
    > $outdir/log.out 2> $outdir/log.err

###

model="threemass"

outdir="../../data/inference/strong/multimass/${data}/${model}-var-${var}"
mkdir -p $outdir

echo $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u multimass-gwpop.py \
    --outdir $outdir \
    --which-data $data \
    --model $model \
    --maximum-uncertainty $var \
    --priors ./priors/threemass-only-mass.prior
    > $outdir/log.out 2> $outdir/log.err