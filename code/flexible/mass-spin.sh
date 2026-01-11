#!/bin/sh

#SBATCH --job-name=6d
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 01-00:00

parentdir="../../data/inference/pixelpop/hmc/mass-spin"
name="nb20x20x5x5x5x5-var1-margsig"

outdir="${parentdir}/${name}"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u mass-spin.py \
    --name $name \
    --parentdir $parentdir \
    --marginalize-sigma \
    --maximum-variance 1 \
    --parallel 1 \
    --seed 1 \
    > "$outdir/log.out" 2> "$outdir/log.err"
