import os
import sys
from copy import deepcopy

import jax
jax.config.update('jax_enable_x64', True)

import h5ify
import numpy as np
from tqdm import trange

from vt import get_inj_priors
from vt import draw_injection

model = dict(
    mass_1_source='highpass_broken_powerlaw_two_peaks',
    mass_ratio='highpass_powerlaw',
    redshift='powerlaw',
    a_1='iid_truncnorm',
    a_2='iid_truncnorm',
    cos_tilt='iso_gauss',
)


def list_to_dict(arr):
    return {
        k : np.concatenate([np.atleast_1d(a[k]) for a in arr])
        for k in arr[0].keys()
    }


if __name__ == '__main__':
    #outdir = '/n/home03/newolfe/projects/tilts-and-kicks/data/inference/injection'
    #outdir = f'{outdir}/tests/260207/nobs70-seed746566-ulin-broad'
    outdir = os.path.abspath(sys.argv[1])
    posterior = h5ify.load(f'{outdir}/extras.h5')

    nsamples = len(posterior['variance'])
    nmc = 100

    all_injs = []

    for i in trange(nsamples):
        parameters = {k : v[i] for k, v in posterior.items()}
        parameters['z_max'] = 1.45
        parameters['lam_2'] = 1 - parameters['lam_1'] - parameters['lam_0']

        priors = get_inj_priors(model, parameters)

        injs = []

        for j in trange(nmc):
            inj = draw_injection(deepcopy(priors), model, parameters)
            injs.append(inj)

        all_injs.append(list_to_dict(injs))

    all_injs = list_to_dict(all_injs)
    all_injs = {k : v.reshape(nsamples, nmc) for k, v in all_injs.items()}

    h5ify.save(f'{outdir}/mcppd.h5', all_injs, mode='w')
