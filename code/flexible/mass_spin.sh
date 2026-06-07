#!/bin/sh

#SBATCH --job-name=6d
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 01-00:00

parentdir="../../data/inference/pixelpop/hmc/mass-spin"
name="nb20x20x5x5x5x5-var1-margsig-nwarm5e5"

outdir="${parentdir}/${name}"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u mass_spin.py \
    --name $name \
    --parentdir $parentdir \
    --bins 20 20 5 5 5 5 \
    --marginalize-sigma \
    --maximum-variance 1 \
    --parallel 1 \
    --seed 1 \
    > "$outdir/log.out" 2> "$outdir/log.err"
