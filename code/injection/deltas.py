import os
import json
from copy import deepcopy
from argparse import ArgumentParser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import h5ify
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from jax_tqdm import scan_tqdm

from bilby import run_sampler
from bilby.core.prior import Uniform
from bilby.core.prior import ConditionalPriorDict

from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

from likelihood import get_bilby_likelihood

from models import log_powerlaw_redshift
from models import log_iid_spin_mag_truncnorm
from models import log_nid_iso_gauss_tilt
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth

nobs = 150
maximum_variance = 1

parser = ArgumentParser()
parser.add_argument('--path', help='path to vt file')
parser.add_argument('--truths', help='path to json file with true parameters')


def parse_args():
    """ return command-line arguments """
    args = parser.parse_args()
    with open(args.truths, 'r') as f:
        truths = json.loads(f.read())
        truths = {k : np.asarray(v).item() for k, v in truths.items()}
    return args.path, truths


def log_model(dataset, parameters):
    """ log-density of the population model """
    if 'lam_2' not in parameters:
        lam_2 = 1 - parameters['lam_1'] - parameters['lam_0']
    else:
        lam_2 = parameters['lam_2']

    log_p_m1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
        dataset['mass_1'],
        alpha_1=parameters['alpha_1'],
        alpha_2=parameters['alpha_2'],
        mlow_1=parameters['mlow_1'],
        break_mass=parameters['break_mass'],
        delta_m_1=parameters['delta_m_1'],
        lam_fractions=(
            parameters['lam_0'], parameters['lam_1'], lam_2
        ),
        mpp_1=parameters['mpp_1'],
        sigpp_1=parameters['sigpp_1'],
        mpp_2=parameters['mpp_2'],
        sigpp_2=parameters['sigpp_2'],
        mmax=300.0,
        gaussian_mass_maximum=100.0
    )

    log_p_q = PowerlawPlusPeak_MassRatio(
        dataset,
        slope=parameters['beta'],
        minimum=parameters['mlow_1'],
        delta_m=parameters['delta_m_1']
    )

    log_p_z = log_powerlaw_redshift(dataset, parameters)

    log_p_chi = log_iid_spin_mag_truncnorm(dataset, parameters)

    parameters['sigma_spin'] = jnp.exp(parameters['log_sigma_spin'])
    parameters['mu_spin'] = jnp.exp(parameters['log_mu_spin'])
    log_p_tau = log_nid_iso_gauss_tilt(dataset, parameters)

    return log_p_m1 + log_p_q + log_p_z + log_p_chi + log_p_tau


def get_data(key, nobs, path):
    """ load in detected injections; format for pdet estimation and delta
        posteriors.
    """
    detectable = h5ify.load(path)
    ndet = len(detectable['log_prior'])

    print('ndet =', ndet)
    print('ndet / ntot =', ndet / detectable['total_generated'])

    detectable['mass_1'] = detectable.pop('mass_1_source')

    #print(detectable['parameters'])

    idxs = jax.random.choice(
        key,
        ndet,
        shape=(nobs,),
        replace=False
    )
    idxs = jnp.sort(idxs)

    posteriors = deepcopy(detectable)
    posteriors.pop('attrs')
    posteriors.pop('total_generated')
    posteriors.pop('model')
    posteriors.pop('parameters')
    posteriors = {k : v[idxs].reshape(nobs, 1) for k, v in posteriors.items()}

    return posteriors, detectable


def taper(v):
    """ its actually a hard cut! """
    return jnp.nan_to_num(-1e10 * (v >= maximum_variance), nan=0)


if __name__ == '__main__':
    path, truths = parse_args()
    truths['log_sigma_spin'] = np.log(truths.pop('sigma_spin')).item()
    truths['log_mu_spin'] = np.log(truths.pop('mu_spin')).item()
    print(truths)

    posteriors, injections = get_data(jax.random.key(42), nobs, path)

    likelihood = get_bilby_likelihood(
        log_model,
        posteriors,
        injections,
        taper=taper,
        rate=False
    )

    print('lnl at truths: ', likelihood.log_likelihood(parameters=truths))
    print('extras at truths: ', likelihood.generate_extra_statistics(truths))

    priors = ConditionalPriorDict('./priors/lvk-lnsigma.prior')

    # tight prior or we run into variance issues
    priors['log_sigma_spin'].minimum = np.log(1e-2 * 0.5).item()  #np.log(1e-3 * 0.5).item()
    priors['log_sigma_spin'].maximum = np.log(1e-2 * 1.5).item()  #np.log(1e-3 * 1.5).item()

    # tight prior here as well, for testing
    priors.pop('mu_spin')
    priors['log_mu_spin'] = Uniform(
        minimum=np.log(0.9).item(),
        maximum=0.0
    )

    outdir = '../../data/runs/tests/260207'

    result = run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        sampler='dynesty',
        sample='acceptance-walk',
        naccept=5,
        nlive=250,
        # need enough live points to resolve w/in and w/o variance cut ...
    )

    posterior = result.posterior.to_dict('list')
    posterior = {k : jnp.array(v) for k, v in posterior.items()}

    n = len(result.posterior)

    @scan_tqdm(n)
    def step(carry, d):
        _, x = d
        return carry, likelihood.generate_extra_statistics(x)

    _, extras = jax.lax.scan(step, None, (jnp.arange(n), posterior))
    posterior = dict(**posterior, **extras)
    h5ify.save(f'{outdir}/extras.h5', posterior, mode='w')
