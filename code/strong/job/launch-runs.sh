#!/bin/bash
set -euo pipefail

cd ..

submit() {
    local config
    config=$(realpath "$1")
    (source "$config"; mkdir -p "$outdir"
     sbatch \
         --job-name="$(basename "$config" .sh)" \
         --output="$outdir/slurm-%j.out" \
         --error="$outdir/slurm-%j.err" \
         --export=ALL,CONFIG="$config" \
         "./submit-multimass-gwpop.sh"
    )
}

submit configs/bpl2p-v1.sh
# submit configs/threemass-v1.sh
