#!/bin/sh

#SBATCH --job-name=m1m2t1t2
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH -t 03-00:00

parentdir="../../data/inference/pixelpop/hmc/m1m2t1t2"
name="test"

outdir="${parentdir}/${name}"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/seqpop/bin/python -u m1m2t1t2.py \
    --name $name \
    --parentdir $parentdir \
    > "$outdir/log.out" 2> "$outdir/log.err"
