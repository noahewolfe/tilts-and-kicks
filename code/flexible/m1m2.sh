#!/bin/sh

#SBATCH --job-name=m1m2
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 03-00:00

parentdir="../../data/inference/pixelpop/hmc/m1m2"
name="nb10-var1"

outdir="${parentdir}/${name}"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u m1m2.py \
    --name $name \
    --parentdir $parentdir \
    --nbins 10 \
    --maximum-variance 1 \
    > "$outdir/log.out" 2> "$outdir/log.err"
