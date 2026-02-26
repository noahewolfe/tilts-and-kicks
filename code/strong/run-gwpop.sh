#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export JAX_ENABLE_X64=1

data="stegmann"
model="stegmann"
var="inf"
seed=1701

outdir="../../data/inference/strong/gwpop-tests"
outdir="${outdir}/model-${model}_data-${data}_var-${var}_seed-${seed}-rerun"

mkdir -p $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u spl-gwpop.py \
    --outdir $outdir \
    --which-data $data \
    --which-model $model \
    --sampling-seed $seed \
    --maximum-uncertainty $var \
    > $outdir/log.out 2> $outdir/log.err
