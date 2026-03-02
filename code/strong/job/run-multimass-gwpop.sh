#!/bin/bash
set -euo pipefail

CONFIG=${1:?Usage: run-multimass-gwpop.sh <config.sh>}
source "$CONFIG"

export JAX_ENABLE_X64=1
unset PYTHONPATH

PYTHON=/work/submit/newolfe/miniforge3/envs/just-for-kicks-260228/bin/python

ARGS=(
    --outdir              "$outdir"
    --which-data          "$which_data"
    --model               "$model"
    --nlive               "$nlive"
    --maximum-uncertainty "$max_uncertainty"
    --sampling-seed       "$sampling_seed"
    --sampler-settings    "$sampler_settings"
    --priors              "$priors"
    --constrain-mu-order  "${constrain_mu_order:-none}"
)
[[ "${stable_expit:-false}" == "true" ]] && ARGS+=(--stable-expit)

exec "$PYTHON" -u multimass-gwpop.py "${ARGS[@]}"
