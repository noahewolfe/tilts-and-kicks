#!/bin/sh

#SBATCH -c 1
#SBATCH -p submit
#SBATCH --mem=10GB
#SBATCH -t 00-05:00
#SBATCH --output slurm_%j.out
#SBATCH --error slurm_%j.err

outdir="../../data/inference/strong/multimass/noah/bpl2p/log-sigma/var1/nlive/10000"
model="default-spin-bpl2p-mass"

#outdir="../../data/inference/strong/multimass/noah/twomass/sampling_seed"
#model="twomass"

#outdir="../../data/inference/strong/multimass/noah/threemass/sampling_seed"
#model="threemass"

/work/submit/newolfe/miniforge3/envs/just-for-kicks-260228/bin/python -u ppd.py \
	--result $outdir/run_result.hdf5 \
	--outdir $outdir \
	--model $model \
	--stable-expit \
	--sample-log-sigma \
	> $outdir/ppd.out 2> $outdir/ppd.err

#python -u ppd.py \
#	--result ../../data/inference/strong/multimass/noah/twomass/sampling_seed/merged.h5 \
#	--outdir ../../data/inference/strong/multimass/noah/twomass/sampling_seed \
#	--model twomass \
#	--stable-expit

#python -u ppd.py \
#	--result ../../data/inference/strong/multimass/noah/bpl2p/sampling_seed/merged.h5 \
#	--outdir ../../data/inference/strong/multimass/noah/bpl2p/sampling_seed \
#	--model default-spin-bpl2p-mass \
#	--stable-expit
