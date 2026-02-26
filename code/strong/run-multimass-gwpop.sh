#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.75
export CUDA_VISIBLE_DEVICES=1
export JAX_ENABLE_X64=1

data="noah"
#model="default-spin-simple-power-law-mass"
model="default-spin-bpl2p-mass"
#model="twomass"
#model="threemass"
nlive=1000
var=1

outdir="../../data/inference/strong/multimass/${data}/${model}-var-${var}-nlive${nlive}-stableexpit"
mkdir -p $outdir

echo $outdir

$HOME/.conda/envs/just-for-kicks/bin/python -u multimass-gwpop.py \
    --outdir $outdir \
    --which-data $data \
    --model $model \
    --maximum-uncertainty $var \
    --nlive $nlive \
    --stable-expit \
    --priors ./priors/onemass-only-mass.prior \
    > $outdir/log.out 2> $outdir/log.err

status=$?
if command -v mail >/dev/null 2>&1; then
  if [ $status -eq 0 ]; then
    echo "SUCCESS: completed on $(hostname) at $(date)" \
      | mail -s "multimass-gwpop ${model} SUCCESS" noah.wolfe@ligo.org
  else
    echo "FAIL ($status): on $(hostname) at $(date)" \
      | mail -s "multimass-gwpop ${model} FAIL ($status)" noah.wolfe@ligo.org
  fi
fi
exit $status
