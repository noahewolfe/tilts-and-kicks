#!/bin/bash

set -euox pipefail

export CUDA_VISIBLE_DEVICES=0

seed=1701
numsamples=1000
numwarmup=10000
thinning=10
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
    #> "${outdir}/log.out" 2> "${outdir}/log.err"
