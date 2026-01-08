#!/bin/sh

#SBATCH --job-name=m1m2
#SBATCH -p iaifi_gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 07-00:00

parentdir="../../data/inference/pixelpop/hmc/m1m2"
name="nb10-var1-margsig"

outdir="${parentdir}/${name}"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/just-for-kicks/bin/python -u m1m2.py \
    --name $name \
    --parentdir $parentdir \
    --nbins 10 \
    --marginalize-sigma \
    --maximum-variance 1 \
    > "$outdir/log.out" 2> "$outdir/log.err"
