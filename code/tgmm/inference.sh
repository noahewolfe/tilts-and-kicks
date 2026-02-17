#!/bin/bash

set -euox pipefail

export CUDA_VISIBLE_DEVICES=3

seed=1706
numsamples=1000
numwarmup=100000
thinning=1
label="o4a-mass-model"

outdir="../../data/inference/tgmm"
outdir="${outdir}/${label}-seed${seed}-nsamp${numsamples}-nwarm${numwarmup}-thin${thinning}"
mkdir -p $outdir

python -u inference.py \
    --outdir $outdir \
    --seed $seed \
    --num-samples $numsamples \
    --num-warmup $numwarmup \
    --thinning $thinning \
    > "${outdir}/log.out" 2> "${outdir}/log.err"
