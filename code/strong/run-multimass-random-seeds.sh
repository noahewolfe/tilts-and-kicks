#!/bin/bash

# trying multiple random sampling seeds to see if that changes
# results with my dataset

export CUDA_VISIBLE_DEVICES=3
export JAX_ENABLE_X64=1


data="noah"
var=1
sample="fast"
nlive=100
model="default-spin-simple-power-law-mass"

parentdir="../../data/inference/strong/multimass/${data}/seeds"

###

for seed in $(seq 1702 1711)
    outdir="${parentdir}/${model}-var-${var}-seed${seed}"
    mkdir -p $outdir

    $HOME/.conda/envs/just-for-kicks/bin/python -u multimass-gwpop.py \
        --outdir $outdir \
        --which-data $data \
        --model $model \
        --maximum-uncertainty $var \
        --sampler-settings $sample \
        --nlive $nlive \
        --sampling-seed $seed \
        > $outdir/log.out 2> $outdir/log.err