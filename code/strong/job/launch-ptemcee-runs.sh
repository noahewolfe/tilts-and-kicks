#!/bin/bash
set -euo pipefail

# Submit three independent ptemcee-hybrid inference runs from code/strong/.
# Production settings override the express/1h defaults in submit-ptemcee.sh.

BASE="../../data/inference/strong/multimass/noah"

sbatch --partition=submit-gpu --time=48:00:00 \
  --export=ALL,\
"MODEL=default-spin-simple-power-law-mass",\
"PRIORS=./priors/onemass-only-mass.prior",\
"NEST_RESULT=${BASE}/default/nlive1000-var1-stableexpit/run_result.hdf5",\
"OUTDIR=${BASE}/default/ptemcee-hybrid-var5",\
"MAX_UNCERTAINTY=5",\
"NWALKERS=200" \
  ./job/submit-ptemcee.sh

exit 0

sbatch --partition=submit-gpu --time=48:00:00 \
  --export=ALL,\
"MODEL=twomass",\
"PRIORS=./priors/twomass-only-mass.prior",\
"NEST_RESULT=${BASE}/twomass/constrain_mu_order/ascending/run_result.hdf5",\
"OUTDIR=${BASE}/twomass/ptemcee-hybrid-var1",\
"MAX_UNCERTAINTY=1",\
"NWALKERS=200" \
  ./job/submit-ptemcee.sh

sbatch --partition=submit-gpu --time=48:00:00 \
  --export=ALL,\
"MODEL=threemass",\
"PRIORS=./priors/threemass-only-mass.prior",\
"NEST_RESULT=${BASE}/threemass/sampling_seed/1703/run_result.hdf5",\
"OUTDIR=${BASE}/threemass/ptemcee-hybrid-var1",\
"MAX_UNCERTAINTY=1",\
"NWALKERS=200" \
  ./job/submit-ptemcee.sh
