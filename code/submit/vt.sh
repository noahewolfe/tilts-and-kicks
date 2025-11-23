#!/bin/bash
set -euox pipefail

# HTCondor passes the array index as the first argument
TASK_ID=${1:?need task id}

outdir="$HOME/projects/tilts-and-kicks/data/vt/test"
initseed=10300
seed=$((initseed + TASK_ID))
outdir="${outdir}/${seed}"

echo "$outdir"
mkdir -p "$outdir"

py="$HOME/.conda/envs/seqpop/bin/python"
"$py" vt.py \
    --outdir "$outdir" \
    --ninj 1 \
    --snr-threshold 11 \
    --seed "$seed" \
    --parameters "./parameters/vt-o4a-strong-maxl-iso-tilts.json" \
    > "${outdir}/log.out" 2> "${outdir}/log.err"
