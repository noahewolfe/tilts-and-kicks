#!/bin/bash
set -euo pipefail

CONFIG=${1:?Usage: run-massbinned-gwpop.sh <config.sh>}
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
    --bin-edges           "$bin_edges"
)
[[ "${dynamic:-false}" == "true" ]] && ARGS+=(--dynamic)

exec "$PYTHON" -u massbinned-gwpop.py "${ARGS[@]}"
