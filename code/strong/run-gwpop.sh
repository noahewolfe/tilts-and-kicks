#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export JAX_ENABLE_X64=1

#outdir='../../data/inference/strong/Gaussian_Isotropic_Cut-Stegmann-model_Stegmann-code_Noah-data'
outdir='../../data/inference/strong/Gaussian_Isotropic_Cut-Noah-model_Stegmann-code_Noah-data'
mkdir -p $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u spl-gwpop.py \
    > $outdir/log.out 2> $outdir/log.err
