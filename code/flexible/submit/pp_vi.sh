#!/bin/sh

#SBATCH --job-name=jfk-ct1-ct2
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=100GB
#SBATCH -t 03-00:00

set -euox pipefail

export CUDA_VISIBLE_DEVICES=3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

cd $HOME/projects/tilts-and-kicks/code/flexible

outdir="../../data/inference/pixelpop/vi/ct1-ct2-nb40-s1e6-b1e2"

mkdir -p $outdir

$HOME/.conda/envs/seqpop/bin/python -u pp_vi.py \
    --outdir $outdir \
    --flow 'bnaf' \
    --flow-kwargs '{"nn_block_dim" : 64}' \
    --train-kwargs '{"steps" : 1000000, "batch_size" : 100, "lr" : 0.1, "final_lr": 0}' \
    --nbins 40 \
    --maximum-variance 1 \
    --parameters '{"log_mass_1" : [1.0986122886681098, 5.703782474656201], "cos_tilt_1" : [-1.0, 1.0], "cos_tilt_2" : [-1.0, 1.0]}' \
    > "$outdir/slurm.out" 2> "$outdir/slurm.err"
