#python cqueue.py vt.py --submit --parentdir ../../data/vt/fly-extra --queue 1000 --args '--ninj 1000 --snr-threshold 11 --init-seed 1100 --model ./parameters/model.json --parameters ./parameters/o4a-strong-maxl.json'

export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME="cpu"

python -u vt.py \
    --outdir './test' \
    --ninj 2 \
    --index 0 \
    --init-seed 1 \
    --snr-threshold 11 \
    --model ./parameters/model.json \
    --parameters ./parameters/vt-o4a-strong-maxl-sharp-peak.json \
    --injection-waveform-approximant 'IMRPhenomXP' \
    --recovery-waveform-approximant 'IMRPhenomXP' \
