#!/bin/bash
set -euox pipefail

proj="$HOME/projects/tilts-and-kicks"

# HTCondor passes the array index as the first argument
TASK_ID=${1:?need task id}

outdir="$proj/data/vt/mock"
initseed=1
seed=$((initseed + TASK_ID))
outdir="${outdir}/${seed}"

echo "$outdir"
mkdir -p "$outdir"

# .py currently assumes we're here
cd "$proj/code"

py="$HOME/.conda/envs/seqpop/bin/python"
"$py" -u vt.py \
    --outdir "$outdir" \
    --ninj 50 \
    --snr-threshold 11 \
    --seed "$seed" \
    --model '{"cos_tilt" : "iso_gauss"}' \
    --parameters "./parameters/o4a-strong-maxl.json" \
    > "${outdir}/log.out" 2> "${outdir}/log.err"
