#!/bin/bash
set -euo pipefail

CONFIG=${1:?Usage: run-identifiable-gwpop.sh <config.sh>}
source "$CONFIG"

export JAX_ENABLE_X64=1
unset PYTHONPATH

PYTHON=/work/submit/newolfe/miniforge3/envs/just-for-kicks-260228/bin/python

ARGS=(
    --outdir              "$outdir"
    --which-data          "$which_data"
    --nlive               "$nlive"
    --maximum-uncertainty "$max_uncertainty"
    --sampling-seed       "$sampling_seed"
    --sampler-settings    "$sampler_settings"
    --priors              "$priors"
    --constrain-mu-order  "${constrain_mu_order:-none}"
)
[[ "${sample_log_sigma:-false}" == "true" ]] && ARGS+=(--sample-log-sigma)
[[ "${dynamic:-false}" == "true" ]] && ARGS+=(--dynamic)

exec "$PYTHON" -u identifiable-gwpop.py "${ARGS[@]}"
