export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME="cpu"

python -u inference.py \
    --outdir ../../data/inference/injection/test \
    --posteriors ../../data/pe/tests/260207/posteriors.hdf5 \
    --injections ../../data/vt/tests/260213/detectable.hdf5 \
    --truths ./parameters/astro-o4a-strong-maxl-sigma-spin-1e-2-xi-0p3.json \
    --seed 42