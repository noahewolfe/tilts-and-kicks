#!/bin/sh

#SBATCH --job-name=hmc
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=50GB
#SBATCH -t 03-00:00

parentdir="../../data/inference/pixelpop"
name="hmc/test"

outdir="${parentdir}/${name}"

mkdir -p $outdir

/n/home03/newolfe/.conda/envs/seqpop/bin/python -u hmc.py \
    --name $name \
    --parentdir $parentdir \
    --warmup 100 \
    --tot-samples 1000 \
    --parallel 1 \
    --thinning 500 \
    --nbins 10 \
    --binned-parameters '{"log_mass_1" : [1.0986122886681098, 5.703782474656201], "cos_tilt_1" : [-1.0, 1.0], "cos_tilt_2" : [-1.0, 1.0]}' \
    #> "$outdir/log.out" 2> "$outdir/log.err"
