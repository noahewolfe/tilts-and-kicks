#!/bin/bash
set -euo pipefail

# Submit three independent ptemcee-hybrid inference runs from code/strong/.
# Production settings override the express/1h defaults in submit-ptemcee.sh.

BASE="../../data/inference/strong/multimass/noah"

sbatch --partition=submit-gpu --time=48:00:00 \
  --export=ALL,\
"MODEL=default-spin-bpl2p-mass",\
"PRIORS=./priors/onemass-only-mass.prior",\
"NEST_RESULT=${BASE}/bpl2p-v1-nlive1000/sampling_seed/1702/run_result.hdf5",\
"OUTDIR=${BASE}/bpl2p-v1-nlive1000/ptemcee-hybrid",\
"NWALKERS=200" \
  ./job/submit-ptemcee.sh

sbatch --partition=submit-gpu --time=48:00:00 \
  --export=ALL,\
"MODEL=twomass",\
"PRIORS=./priors/twomass-only-mass.prior",\
"NEST_RESULT=${BASE}/twomass/constrain_mu_order/ascending/run_result.hdf5",\
"OUTDIR=${BASE}/twomass/ptemcee-hybrid",\
"NWALKERS=200" \
  ./job/submit-ptemcee.sh

sbatch --partition=submit-gpu --time=48:00:00 \
  --export=ALL,\
"MODEL=threemass",\
"PRIORS=./priors/threemass-only-mass.prior",\
"NEST_RESULT=${BASE}/threemass/sampling_seed/1703/run_result.hdf5",\
"OUTDIR=${BASE}/threemass/ptemcee-hybrid",\
"NWALKERS=200" \
  ./job/submit-ptemcee.sh
