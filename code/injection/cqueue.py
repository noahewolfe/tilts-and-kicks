import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('script', type=str)
parser.add_argument('--parentdir', type=str)
parser.add_argument('--request-cpus', type=int, default=1)
parser.add_argument('--request-memory', type=str, default='4 GB')
parser.add_argument('--request-disk', type=str, default='1 GB')
parser.add_argument('--queue', type=int, default=1)
parser.add_argument('--submit', action='store_true')
parser.add_argument(
    '--avx',
    action='store_true',
    help='require CPUs with AVX/AVX2 instruction sets'
)
parser.add_argument('--args', type=str)

environment = (
    'JAX_ENABLE_X64=True; '
    'JAX_PLATFORMS=cpu; '
    'JAX_CUDA_VISIBLE_DEVICES=; '
    'JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_$(Cluster)_$(Process)'
)

if __name__ == '__main__':
    args = parser.parse_args()

    initialdir = os.getcwd()
    parentdir = args.parentdir
    queue = args.queue

    for i in range(queue):
        os.makedirs(f'{parentdir}/{i}', exist_ok=True)

    outdir = f'{parentdir}/$(Process)'

    arguments = f'-u {args.script} --outdir {outdir}'
    arguments = fr'{arguments} --index $(Process) {args.args}'

    requirements = '(Microarch >= "x86_64-v3")' if args.avx else ''

    submit = fr"""
universe         = vanilla
executable       = /home/noah.wolfe/.conda/envs/just-for-kicks/bin/python
accounting_group = ligo.sim.o4.cbc.bayesianpopulations.parametric

initialdir       = {initialdir}

arguments        = {arguments}

environment      = {environment}
getenv           = True

requirements     = {requirements}
request_cpus     = {args.request_cpus}
request_memory   = {args.request_memory}
request_disk     = {args.request_disk}

output          = {outdir}/run.$(Cluster).out
error           = {outdir}/run.$(Cluster).err
log             = {outdir}/run.$(Cluster).log

queue {queue}
"""

    path = f'{parentdir}/job.submit'

    with open(path, 'w') as f:
        f.write(submit)

    if args.submit:
        subprocess.run(['condor_submit', path])
    else:
        print(path)
