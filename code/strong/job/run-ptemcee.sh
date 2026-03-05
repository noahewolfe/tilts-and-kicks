#!/bin/bash
set -euo pipefail

export JAX_ENABLE_X64=1
unset PYTHONPATH

OUTDIR=${OUTDIR:-./test-ptemcee}
MODEL=${MODEL:-twomass}
PRIORS=${PRIORS:-./priors/twomass-only-mass.prior}
NEST_RESULT=${NEST_RESULT:-../../data/inference/strong/multimass/noah/twomass/constrain_mu_order/ascending/run_result.hdf5}
NWALKERS=${NWALKERS:-200}
NEST_RESULT_MU_ORDER=${NEST_RESULT_MU_ORDER:-ascending}
MAX_UNCERTAINTY=${MAX_UNCERTAINTY:-inf}
SEED=${SEED:-746566}

mkdir -p "$OUTDIR"

/work/submit/newolfe/miniforge3/envs/just-for-kicks-260228/bin/python -u ptemcee-multimass-gwpop.py \
    --outdir "$OUTDIR" \
    --which-data 'noah' \
    --model "$MODEL" \
    --priors "$PRIORS" \
    --nest-result "$NEST_RESULT" \
    --nwalkers "$NWALKERS" \
    --nest-result-mu-order "$NEST_RESULT_MU_ORDER" \
    --sampling-seed "$SEED" \
    --maximum-uncertainty "$MAX_UNCERTAINTY" \
    --stable-expit \
    > "$OUTDIR/log.out" 2> "$OUTDIR/log.err"
