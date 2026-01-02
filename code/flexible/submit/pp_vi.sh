#!/bin/sh

#SBATCH --job-name=jfk-ct1-ct2
#SBATCH -p iaifi_gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=100GB
#SBATCH -t 03-00:00

set -euox pipefail

cd $HOME/projects/tilts-and-kicks/code/flexible

outdir="../../data/inference/pixelpop/vi/test-m2gap"

mkdir -p $outdir

$HOME/.conda/envs/seqpop/bin/python -u pp_vi.py \
    --outdir $outdir \
    --flow 'bnaf' \
    --flow-kwargs '{"nn_block_dim" : 8}' \
    --train-kwargs '{"steps" : 10, "batch_size" : 10, "lr" : 0.0001, "final_lr": 0.0001}' \
    --nbins 10 \
    --maximum-variance 1 \
    --parameters '{"log_mass_1" : [1.0986122886681098, 5.703782474656201], "cos_tilt_1" : [-1.0, 1.0], "cos_tilt_2" : [-1.0, 1.0]}' \
    --prior ./priors/m2gap.prior \
    --model-in-m2 \
    #> "$outdir/slurm.out" 2> "$outdir/slurm.err"
