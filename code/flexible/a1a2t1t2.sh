#!/bin/sh

#SBATCH --job-name=a1a2t1t2
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 01-00:00

parentdir="../../data/inference/pixelpop/hmc/a1a2t1t2"
name="nb10x10x10x10-var1-chain4-margsig-nwarm250-mtd10-p65"

outdir="${parentdir}/${name}"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u a1a2t1t2.py \
    --name $name \
    --parentdir $parentdir \
    --marginalize-sigma \
    --maximum-variance 1 \
    --parallel 1 \
    --seed 1705 \
    > "$outdir/log.out" 2> "$outdir/log.err"
