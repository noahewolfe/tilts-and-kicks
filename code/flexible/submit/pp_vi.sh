#!/bin/sh

#SBATCH --job-name=jfk-ct1-ct2
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=100GB
#SBATCH -t 03-00:00

set -euox pipefail

cd /n/home03/newolfe/projects/tilts-and-kicks/code/flexible

outdir="../../data/inference/pixelpop/vi/ct1-ct2-nb10-s1e6-b1e1"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/seqpop/bin/python -u pp_vi.py \
    --outdir $outdir \
    --flow 'bnaf' \
    --flow-kwargs '{"nn_block_dim" : 64}' \
    --train-kwargs '{"steps" : 1000000, "batch_size" : 10, "lr" : 0.1, "final_lr": 0}' \
    --nbins 10 \
    --maximum-variance 1 \
    --parameters '{"cos_tilt_1" : [-1.0, 1.0], "cos_tilt_2" : [-1.0, 1.0]}' \
    > "$outdir/slurm.out" 2> "$outdir/slurm.err"
