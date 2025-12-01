#!/bin/sh

#SBATCH --job-name=pp-vi
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=100GB
#SBATCH -t 03-00:00

set -euox pipefail

cd /n/home03/newolfe/projects/tilts-and-kicks/code

outdir="../data/inference/pixelpop/vi/test-ct1-ct2"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/seqpop/bin/python -u pp_vi.py \
    --outdir $outdir \
    --flow 'bnaf' \
    --flow-kwargs '{"nn_block_dim" : 8}' \
    --train-kwargs '{"steps" : 10, "batch_size" : 1, "lr" : 0.001, "final_lr": 0.001}' \
    --nbins 10 \
    --maximum-variance 1 \
    --parameters '{"cos_tilt_1" : [-1.0, 1.0], "cos_tilt_2" : [-1.0, 1.0]}' \
    
    #--parameters '{"log_mass_1" : [1.0986122886681098, 5.703782474656201], "redshift" : [0, 1.45]}'
    #> "$outdir/slurm.out" 2> "$outdir/slurm.err"
