#!/bin/bash
set -euo pipefail

config=$(realpath "${1:?Usage: launch-array-identifiable.sh <config.sh> <param> <val1> [val2 ...]}")
param="${2:?Usage: launch-array-identifiable.sh <config.sh> <param> <val1> [val2 ...]}"
shift 2
values=("$@")
[[ ${#values[@]} -ge 1 ]] || { echo "Error: at least one value required" >&2; exit 1; }

source "$config"   # get base $outdir

_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_dir/.."   # code/strong/

mkdir -p "$outdir"
for v in "${values[@]}"; do
    mkdir -p "${outdir}/${param}/${v}"
done

n=${#values[@]}
values_str=$(IFS=':'; echo "${values[*]}")

sbatch \
    --job-name="$(basename "$config" .sh)-${param}" \
    --array="0-$((n - 1))" \
    --output="${outdir}/slurm-%A_%a.out" \
    --error="${outdir}/slurm-%A_%a.err" \
    --export=ALL,BASE_CONFIG="$config",SWEEP_PARAM="$param",SWEEP_VALUES="$values_str" \
    ./job/submit-array-task-identifiable.sh
