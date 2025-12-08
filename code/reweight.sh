#!/bin/bash
set -euox pipefail

idx=0

python -u pe.py \
    --outdir "../data/pe/mock/xphm/${idx}" \
    --index $idx \
    --npool 8 \
    --prior-path './priors/bbh.prior' \
    --time-reference 'H1' \
    --nlive 250 \
    --catalog-path '../data/vt/mock/cat.hdf5' \
    --injection-waveform 'xphm' \
    --recovery-waveform 'xphm' \
    --reweight \
    --overwrite \
